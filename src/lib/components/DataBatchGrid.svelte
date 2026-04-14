<script lang="ts">
	import type { DataBatch } from '$lib/DataSource';

	let { dataBatch }: { dataBatch: DataBatch } = $props();
</script>

{dataBatch.xs}

{dataBatch.ys}
