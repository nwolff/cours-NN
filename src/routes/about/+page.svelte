<script>
	const repoUrl = '[GITHUB_SERVER_URL]/[GITHUB_REPOSITORY]';
	const repoName = '[GITHUB_REPOSITORY]';
	const commitUrl = '[GITHUB_SERVER_URL]/[GITHUB_REPOSITORY]/commit/[GITHUB_SHA]';
	const shortSha = '[SHORT_SHA]';
	const refName = '[GITHUB_REF_NAME]';
	const timestamp = '[TIMESTAMP]';

	const deployed = repoUrl.startsWith('http');
</script>

<dl>
	<dt class="font-semibold">repo</dt>
	<dd class="ml-3">
		{#if deployed}
			<a class="link" href={repoUrl} target="_blank" rel="noreferrer">{repoName}</a>
		{:else}
			{repoName}
		{/if}
	</dd>
	<dt class="font-semibold">branch</dt>
	<dd class="ml-3">{refName}</dd>
	<dt class="font-semibold">commit</dt>
	<dd class="ml-3">
		{#if deployed}
			<a class="link" href={commitUrl} target="_blank" rel="noreferrer">{shortSha}</a>
		{:else}
			{shortSha}
		{/if}
	</dd>
	<dt class="font-semibold">built</dt>
	<dd class="ml-3">{timestamp}</dd>
</dl>
