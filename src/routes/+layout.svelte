<script lang="ts">
	import '../app.css';
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import Logo from './_Logo.svelte';

	let { children } = $props();
</script>

<div class="navbar px-8 py-0">
	<a href="{base}/">
		<Logo size={40} />
	</a>
	<ul class="menu menu-horizontal">
		<li class="text-lg mx-1">
			<a
				href="{base}/temperature_control"
				class={page.url.pathname.endsWith('/temperature_control') ? 'menu-active' : ''}
			>
				Un neurone
			</a>
		</li>
		<li class="text-lg mx-1">
			<a
				href="{base}/zero_one"
				class={page.url.pathname.endsWith('/zero_one') ? 'menu-active' : ''}
			>
				0 et 1
			</a>
		</li>
		<li class="text-lg mx-1">
			<a
				href="{base}/all_digits"
				class={page.url.pathname.endsWith('/all_digits') ? 'menu-active' : ''}
			>
				Tous les chiffres
			</a>
		</li>
		<li class="text-lg mx-1">
			<a href="{base}/fashion" class={page.url.pathname.endsWith('/fashion') ? 'menu-active' : ''}>
				Mode
			</a>
		</li>
	</ul>
	<a href="{base}/about" class="ml-auto text-2xl text-base-content/30 hover:text-base-content" title="À propos">ⓘ</a>
</div>

<div class="divider m-0 pb-6"></div>

<div class="container px-8">
	{@render children()}
</div>
