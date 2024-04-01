<script lang="ts">
	import 'tailwindcss/tailwind.css';
	import { base } from '$app/paths';
	import { page } from '$app/stores';
	import Logo from './_Logo.svelte';
</script>

<div class="navbar px-8 py-0">
	<a href="{base}/">
		<Logo size={40} />
	</a>
	<ul class="menu menu-horizontal">
		<li class="text-lg mx-1">
			<a href="{base}/predict" class={$page.url.pathname.endsWith('/predict') ? 'active' : ''}>
				Reconnaître
			</a>
		</li>
		<li class="text-lg mx-1">
			<a href="{base}/train" class={$page.url.pathname.endsWith('/train') ? 'active' : ''}>
				Entraîner
			</a>
		</li>
		<li class="text-lg mx-1">
			<a href="{base}/evaluate" class={$page.url.pathname.endsWith('/evaluate') ? 'active' : ''}>
				Evaluer
			</a>
		</li>
	</ul>
</div>

<div class="divider m-0 pb-6"></div>

<div class="container px-8">
	<slot />
</div>
