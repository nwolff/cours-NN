<script lang="ts">
	import type { NetworkStats } from '$lib/Network';

	export let stats: NetworkStats;

	const formatter = Intl.NumberFormat('en', { notation: 'compact' });

	$: formattedNumExamples = formatter.format(stats.samplesSeen);
	$: testAccuracyPercent = Math.floor(stats.testAccuracy * 100) || '';
	$: trainingAccuracyPercent = Math.floor(stats.trainingAccuracy * 100) || '';
</script>

<div class="stats shadow stats-vertical">
	<div class="stat">
		<div class="stat-title"><b>Images vues</b></div>
		<div class="stat-value">{formattedNumExamples}</div>
	</div>
	<div class="stat">
		<div class="stat-title"><b>Précision de test</b></div>
		<div class="tooltip" data-tip="{testAccuracyPercent}%">
			<progress class="progress progress-primary" value={testAccuracyPercent} max="100"> </progress>
		</div>
	</div>
	<div class="stat">
		<div class="stat-title">Précision entraînement</div>
		<div class="tooltip" data-tip="{trainingAccuracyPercent}%">
			<progress class="progress" value={trainingAccuracyPercent} max="100"> </progress>
		</div>
	</div>
</div>
