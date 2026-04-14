<script lang="ts">
	import type { NetworkStats } from '$lib/Network';
	import LossChart from '$lib/components/LossChart.svelte';
	import ConfusionMatrix from '$lib/components/ConfusionMatrix.svelte';

	let { stats }: { stats: NetworkStats } = $props();

	let confusionMatrixSize = $state(140);

	const formatter = Intl.NumberFormat('en', { notation: 'compact' });

	const formattedNumExamples = $derived(formatter.format(stats.samplesSeen));
	const testAccuracyPercent = $derived(Math.floor(stats.test?.accuracy * 100) || '');
	const classes = $derived(stats.test.classes);
	const losses = $derived(stats.losses);
	const labelsAndPredictions = $derived(
		[stats.test.labels, stats.test.predictions] as [number[], number[]]
	);
</script>

<div class="stats shadow bg-base-200 stats-vertical">
	<div class="stat">
		<div class="stat-title">Images vues</div>
		<div class="stat-value">{formattedNumExamples}</div>
	</div>
	{#if losses?.length}
		<div class="stat">
			<div class="stat-title">Perte</div>
			<div class="stat-value">{formatter.format(losses[losses.length - 1].loss)}</div>
			<LossChart {losses} />
		</div>
	{/if}

	{#if testAccuracyPercent}
		<div class="stat">
			<div class="stat-title">Précision de test</div>
			<div class="tooltip" data-tip="{testAccuracyPercent}%">
				<progress class="progress progress-primary" value={testAccuracyPercent} max="100">
				</progress>
			</div>
			<div
				role="complementary"
				class="mt-3"
				onmouseenter={() => (confusionMatrixSize = 280)}
				onmouseleave={() => (confusionMatrixSize = 140)}
			>
				<ConfusionMatrix size={confusionMatrixSize} {classes} {labelsAndPredictions} />
			</div>
		</div>
	{/if}
</div>
