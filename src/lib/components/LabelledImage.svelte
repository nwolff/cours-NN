<script lang="ts">
	import type { DataBatch } from '$lib/DataSource';
	import * as tf from '@tensorflow/tfjs';

	let { dataBatch, indexInBatch = 0 }: { dataBatch: DataBatch; indexInBatch?: number } = $props();

	const image = $derived(dataBatch.xs.slice([indexInBatch], [1]));
	const label = $derived(dataBatch.ys.slice([indexInBatch], [1]));
</script>

<div>
	<dl>
		<dt><b>label</b></dt>
		<dd>
			shape: {label.shape}
			<br />
			{label}
		</dd>
		<dt><b>image</b></dt>
		<dd>
			shape: {image.shape}
			<br />
			mean: {tf.mean(image)}
			<br />
			max: {tf.max(image)}
		</dd>
	</dl>
</div>
