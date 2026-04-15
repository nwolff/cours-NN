import js from '@eslint/js';
import tsPlugin from '@typescript-eslint/eslint-plugin';
import tsParser from '@typescript-eslint/parser';
import svelte from 'eslint-plugin-svelte';
import prettierConfig from 'eslint-config-prettier';
import globals from 'globals';

export default [
	js.configs.recommended,
	...tsPlugin.configs['flat/recommended'],
	...svelte.configs['flat/recommended'],
	prettierConfig,
	...svelte.configs['flat/prettier'],
	{
		languageOptions: {
			globals: {
				...globals.browser,
				...globals.node
			}
		}
	},
	{
		files: ['**/*.svelte'],
		languageOptions: {
			parserOptions: {
				parser: tsParser
			}
		}
	},
	{
		rules: {
			// Allow _-prefixed parameters and variables that are intentionally unused
			'@typescript-eslint/no-unused-vars': [
				'error',
				{ argsIgnorePattern: '^_', varsIgnorePattern: '^_' }
			],
			// Static hrefs in a GitHub Pages project with base path are intentional
			'svelte/no-navigation-without-resolve': 'off'
		}
	},
	{
		ignores: [
			'build/',
			'.svelte-kit/',
			'node_modules/',
			'package-lock.json',
			'.env',
			'.env.*',
			'!.env.example'
		]
	}
];
