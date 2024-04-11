<script lang="ts">
	import type { DataBatch } from '$lib/DataSource';
	import * as tf from '@tensorflow/tfjs';

	export let dataBatch: DataBatch;
	export let indexInBatch: number = 0;

	$: image = dataBatch.xs.slice([indexInBatch], [1]);
	$: label = dataBatch.labels.slice([indexInBatch], [1]);
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
