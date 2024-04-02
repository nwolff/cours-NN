<script lang="ts">
	import type { NetworkStats } from '$lib/models';

	export let stats: NetworkStats;

	const formatter = Intl.NumberFormat('en', { notation: 'compact' });

	$: formattedNumExamples = formatter.format(stats.samplesSeen);
	$: accuracyPercent = Math.floor(stats.accuracy * 100) || '';
</script>

<div class="stats shadow stats-vertical lg:stats-horizontal">
	<div class="stat">
		<div class="stat-title">Images vues</div>
		<div class="stat-value">{formattedNumExamples}</div>
	</div>
	<div class="stat">
		<div class="stat-title">Précision</div>
		<div class="radial-progress" style="--value:{accuracyPercent};" role="progressbar">
			{accuracyPercent}%
		</div>
	</div>
</div>
