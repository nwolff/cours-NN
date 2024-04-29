<script lang="ts">
	import type { NetworkStats } from '$lib/Network';

	export let stats: NetworkStats;

	const formatter = Intl.NumberFormat('en', { notation: 'compact' });

	$: trainingAccuracyPercent = Math.floor(stats.trainingAccuracy * 100) || '';
</script>

<div class="stats shadow bg-base-200 stats-vertical">
	{#if stats.trainingAccuracy}
		<div class="stat">
			<div class="stat-title">Précision entraînement</div>
			<div class="tooltip" data-tip="{trainingAccuracyPercent}%">
				<progress class="progress" value={trainingAccuracyPercent} max="100"> </progress>
			</div>
		</div>
	{/if}
</div>
