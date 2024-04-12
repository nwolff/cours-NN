<script lang="ts">
	// Paint code from https://www.i-am.ai/neural-numbers.html
	import { onMount, onDestroy, createEventDispatcher } from 'svelte';
	import {
		drawImageFitted,
		toGrayscaleInverted,
		blueToGrayscaleInverted,
		findBoundingBox
	} from '$lib/generic/image';
	const dispatch = createEventDispatcher();

	let SCALE_FACTOR = 9;
	let LINEWIDTH = 2 * SCALE_FACTOR;

	let drawCanvas: HTMLCanvasElement;
	let negativeCanvas: HTMLCanvasElement;
	let normalizeCanvas: HTMLCanvasElement;
	let drawContext: CanvasRenderingContext2D;
	let negativeContext: CanvasRenderingContext2D;
	let normalizeContext: CanvasRenderingContext2D;

	let pos = { x: 0, y: 0 };
	let isdown = false;
	let pointerId = -1;

	const eventfunctions: { [id: string]: any } = {
		pointerdown: (e: { clientX: number; clientY: number; pointerId: number }) => {
			if (!isdown) {
				setPosition(e);
				isdown = true;
				pointerId = e.pointerId;
			}
		},
		pointermove: (e: { clientX: number; clientY: number; pointerId: number }) => {
			if (isdown && pointerId === e.pointerId) {
				draw(e);
			}
		},
		pointerup: (e: { pointerId: number }) => {
			if (pointerId === e.pointerId) {
				isdown = false;
				imageReady();
			}
		},
		pointerleave: (e: { pointerId: number }) => {
			if (pointerId === e.pointerId) {
				isdown = false;
			}
		},
		pointercancel: (e: { pointerId: number }) => {
			if (pointerId === e.pointerId) {
				isdown = false;
			}
		}
	};

	onMount(() => {
		addEventListeners();

		normalizeCanvas.width = 28;
		normalizeCanvas.height = 28;

		const updateDimensions = () => {
			SCALE_FACTOR = Math.floor(drawCanvas.clientWidth / 28) - 1;
			LINEWIDTH = 2 * SCALE_FACTOR;
			drawCanvas.width = drawCanvas.clientWidth;
			drawCanvas.height = drawCanvas.clientWidth;
		};
		updateDimensions();
		window.onresize = () => {
			updateDimensions();
		};

		drawContext = drawCanvas.getContext('2d', { willReadFrequently: true })!;
		negativeContext = negativeCanvas.getContext('2d', { willReadFrequently: true })!;
		normalizeContext = normalizeCanvas.getContext('2d', { willReadFrequently: true })!;
		clear();
	});

	onDestroy(() => {
		clear();
		removeEventListeners();
	});

	export function clear() {
		drawContext.fillStyle = 'white';
		drawContext.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
		normalize(100);
	}

	function addEventListeners() {
		for (const eventname in eventfunctions) {
			drawCanvas.addEventListener(eventname, eventfunctions[eventname], {
				passive: true
			});
		}
	}

	function removeEventListeners() {
		for (const eventname in eventfunctions) {
			drawCanvas.removeEventListener(eventname, eventfunctions[eventname]);
		}
	}

	function setPosition(e: { clientX: number; clientY: number }) {
		const rect = drawCanvas.getBoundingClientRect();
		pos.x = e.clientX - rect.left;
		pos.y = e.clientY - rect.top;
	}

	function draw(e: { clientX: number; clientY: number }) {
		const ox = pos.x;
		const oy = pos.y;
		setPosition(e);
		const nx = pos.x;
		const ny = pos.y;
		if (Math.abs(nx - ox) + Math.abs(ny - oy) < 3) {
			pos.x = ox;
			pos.y = oy;
			return;
		}
		drawContext.beginPath(); // begin
		drawContext.lineWidth = LINEWIDTH;
		drawContext.lineCap = 'round';
		drawContext.strokeStyle = 'blue'; // XXX

		drawContext.moveTo(ox, oy); // from
		setPosition(e);
		drawContext.lineTo(nx, ny); // to

		drawContext.stroke(); // draw it!

		normalizeContext.fillStyle = 'black';
		normalizeContext.fillRect(0, 0, normalizeCanvas.width, normalizeCanvas.height);

		normalize(LINEWIDTH);
	}

	// normalize image
	function normalize(SKIPFACTOR: number) {
		let centerx = 0;
		let centery = 0;
		let top = 1000;
		let bottom = -1000;
		let left = 1000;
		let right = -1000;
		let imgData = drawContext.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
		let { data } = imgData;
		blueToGrayscaleInverted(data);
		negativeContext.putImageData(imgData, 0, 0);
		let totalweight = 0;
		for (let i = 0; i < data.length; i += 4 * SKIPFACTOR) {
			const x = (i / 4) % negativeCanvas.width;
			const y = (i / 4 / negativeCanvas.width) | 0;
			totalweight += data[i];
			centerx += data[i] * x;
			centery += data[i] * y;

			if (data[i] > 0) {
				top = Math.min(top, y);
				bottom = Math.max(bottom, y);
				left = Math.min(left, x);
				right = Math.max(right, x);
			}
		}
		if (totalweight > 0) {
			centerx /= totalweight;
			centery /= totalweight;

			const boxsize = Math.max(right - left, bottom - top);

			/*
			According to MNIST normalization:

      		The original black and white (bilevel) images from NIST were size normalized
      		to fit in a 20x20 pixel box while preserving their aspect ratio. The
      		resulting images contain grey levels as a result of the anti-aliasing
      		technique used by the normalization algorithm. the images were centered
      		in a 28x28 image by computing the center of mass of the pixels, and
      		translating the image so as to position this point at the center of the 28x28 field.
      		*/
			normalizeContext.drawImage(
				negativeCanvas,
				left,
				top,
				boxsize,
				boxsize,
				14 + (20 / boxsize) * (left - centerx),
				14 + (20 / boxsize) * (top - centery),
				20,
				20
			);
		} else {
			normalizeContext.fillRect(0, 0, normalizeCanvas.width, normalizeCanvas.height);
		}
	}

	function imageReady() {
		dispatch('imageData', { image: normalizeCanvas });
	}

	//////////////////////////////////////////////
	//
	//  Drop support
	//
	//////////////////////////////////////////////

	let dropHighlighted = false;

	function handleDragOver(event: DragEvent) {
		event.preventDefault();
	}

	function handleDragEnter(_event: DragEvent) {
		dropHighlighted = true;
	}

	function handleDragLeave(_event: DragEvent) {
		dropHighlighted = false;
	}

	function handleDrop(event: DragEvent) {
		event.preventDefault();
		dropHighlighted = false;
		const file = event.dataTransfer?.files?.item(0);
		if (file?.type.startsWith('image/')) {
			const fileURL = URL.createObjectURL(file);
			const image = new Image();
			image.src = fileURL;
			image.onload = () => {
				URL.revokeObjectURL(fileURL);
				drawImageFitted(drawContext, 'white', image);
				const imgData = drawContext.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
				const { data } = imgData;
				toGrayscaleInverted(data);
				negativeContext.putImageData(imgData, 0, 0);
				drawImageFitted(normalizeContext, 'black', negativeCanvas, findBoundingBox(imgData, 20));
				imageReady();
			};
			image.onabort = () => {
				URL.revokeObjectURL(fileURL);
			};
			image.onerror = () => {
				URL.revokeObjectURL(fileURL);
			};
		}
	}
</script>

<div class="drawbox">
	<div class="drawcanvas-wrapper" class:highlighted={dropHighlighted}>
		<canvas
			on:drop={handleDrop}
			on:dragover={handleDragOver}
			on:dragenter={handleDragEnter}
			on:dragleave={handleDragLeave}
			bind:this={drawCanvas}
			class="drawcanvas"
			id="canvas"
			width="140"
			height="140"
		/>
	</div>
	<div class="negativecanvas-wrapper">
		<canvas class="negativecanvas" bind:this={negativeCanvas} width="140" height="140" />
	</div>
	<div class="normalizecanvas-wrapper">
		<canvas class="normalizecanvas" bind:this={normalizeCanvas} width="28" height="28" />
	</div>
</div>

<style>
	.drawbox {
		display: flex;
	}

	.drawbox .drawcanvas-wrapper {
		width: 140px;
		height: 140px;
		border: 8px dashed #666;
		border-radius: 8px;
		box-sizing: content-box;
	}

	.drawbox .drawcanvas-wrapper.highlighted {
		border-color: #00f;
	}

	.drawbox .drawcanvas {
		filter: brightness(1.15);
		pointer-events: all !important;
	}

	.drawbox .negativecanvas-wrapper {
		display: none;
	}

	.drawbox .normalizecanvas-wrapper {
		display: none;
	}
</style>
