<script lang="ts">
	import type { NetworkStats } from '$lib/Network';
	import ConfusionMatrix from '$lib/components/ConfusionMatrix.svelte';

	export let stats: NetworkStats;

	let confusionMatrixSize = 140;

	const formatter = Intl.NumberFormat('en', { notation: 'compact' });

	$: formattedNumExamples = formatter.format(stats.samplesSeen);
	$: testAccuracyPercent = Math.floor(stats.test?.accuracy * 100) || '';
	$: classes = stats.test.classes;
	$: labelsAndPredictions = [stats.test.labels, stats.test.predictions] as [number[], number[]];
</script>

<div class="stats shadow bg-base-200 stats-vertical">
	<div class="stat">
		<div class="stat-title">Images vues</div>
		<div class="stat-value">{formattedNumExamples}</div>
	</div>
	<div class="stat">
		<div class="stat-title">Précision de test</div>
		<div class="tooltip" data-tip="{testAccuracyPercent}%">
			<progress class="progress progress-primary" value={testAccuracyPercent} max="100"> </progress>
		</div>
		<div
			class="mt-3"
			on:mouseenter={() => (confusionMatrixSize = 280)}
			on:mouseleave={() => (confusionMatrixSize = 140)}
		>
			<ConfusionMatrix size={confusionMatrixSize} {classes} {labelsAndPredictions} />
		</div>
	</div>
</div>
