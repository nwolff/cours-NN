import type { Link, Neuron } from './NetworkShape';

export type LinkFilter = (links: Link[]) => Link[];

export const allLinks: LinkFilter = (links) => links;

export function visualWeight(link: Link): number {
	if (link.activation) {
		return link.weight * (1 + 0.5 * link.activation);
	} else {
		return link.weight;
	}
}

export function makeTopNLinksFilter(maxLinks: number) {
	return function (links: Link[]) {
		const length = links.length;
		if (length <= maxLinks) {
			return links;
		}
		const sortedLinks = [...links].sort(
			(l1: Link, l2: Link) => Math.abs(visualWeight(l2)) - Math.abs(visualWeight(l1))
		);
		return sortedLinks.slice(0, maxLinks);
	};
}

function selectAbove(links: Link[], neurons: Neuron[]): Link[] {
	const neuronSet = new Set(neurons);
	console.log(neuronSet.size);
	return links.filter((link) => neuronSet.has(link.b));
}

function selectBelow(links: Link[], neurons: Neuron[]): Link[] {
	const neuronSet = new Set(neurons);
	return links.filter((link) => neuronSet.has(link.a));
}

export const neighborsFilter = farNeighborsFilter;

// Keep all neurons that are directly connected to the given neuron
function immediateNeighborsFilter(neuron: Neuron) {
	return function (links: Link[]) {
		return links.filter((link) => link.a == neuron || link.b == neuron);
	};
}

// Keep all neurons that are somehow connected to the given neuron
function farNeighborsFilter(neuron: Neuron) {
	return function (links: Link[]) {
		const result: Link[] = [];

		// Look up till there is nothing left
		let neurons = [neuron];
		let distance = 1;
		while (neurons.length) {
			const aboveNeighbors = makeTopNLinksFilter(12 * distance ** 1.2)(selectAbove(links, neurons));
			result.push(...aboveNeighbors);
			neurons = aboveNeighbors.map((link) => link.a);
			distance++;
		}

		// Look down till there is nothing left
		distance = 1;
		neurons = [neuron];
		while (neurons.length) {
			const belowNeighbors = makeTopNLinksFilter(12 * distance ** 1.2)(selectBelow(links, neurons));
			result.push(...belowNeighbors);
			neurons = belowNeighbors.map((link) => link.b);
			distance++;
		}

		return result;
	};
}
