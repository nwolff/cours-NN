<script lang="ts">
	import { base } from '$app/paths';
	import { page } from '$app/stores';
	import { Stack, Anchor, AppShell, Navbar, Header } from '@svelteuidev/core';
	import Logo from './_Logo.svelte';
</script>

<AppShell>
	<Navbar slot="navbar" fixed position={{ top: 10, left: 10 }} width={{ base: '100%', sm: 110 }}>
		<Stack>
			<Anchor href="{base}/"><Logo size={40} /></Anchor>
			<Anchor size="lg" href="{base}/predict" underline={$page.url.pathname.endsWith('/predict')}>
				Reconnaître
			</Anchor>
			<Anchor size="lg" href="{base}/train" underline={$page.url.pathname.endsWith('/train')}>
				Entraîner
			</Anchor>
			<Anchor size="lg" href="{base}/evaluate" underline={$page.url.pathname.endsWith('/evaluate')}>
				Evaluer
			</Anchor>
		</Stack>
	</Navbar>

	<!-- Avoid wasted space at top of page -->
	<Header fixed height="-10" />

	<slot />
</AppShell>
