// Models the shape and layout of a dense feed forward neural network

import type { LayerVariable } from '@tensorflow/tfjs';
import { zip2, zip3 } from '$lib/generic/utils';

export class Neuron {
	x: number;
	y: number;
	activation: number;

	constructor(x: number, y: number, activation = 0) {
		this.x = x;
		this.y = y;
		this.activation = activation;
	}
}

export class Link {
	a: Neuron;
	b: Neuron;
	weight: number;

	constructor(a: Neuron, b: Neuron, weight: number) {
		this.a = a;
		this.b = b;
		this.weight = weight;
	}
}

// Describes the layout of a single layer.
export type LayerSpec = {
	name: string;
	neuron_count: number;
	width: number;
	height: number;
	marker_size: number;
	neurons_per_row?: number;
	classes?: string[];
	rotateClassNames?: boolean;
};

// Preforms the layout of a single layer, placing the neurons accordingly
export class Layer {
	name: string;
	marker_size: number;
	neurons: Neuron[];
	classes?: string[];
	rotateClassNames: boolean;

	constructor(layer_start_y: number, spec: LayerSpec) {
		this.name = spec.name;
		this.marker_size = spec.marker_size;
		this.classes = spec.classes;
		this.rotateClassNames = spec.rotateClassNames ? true : false;

		const neurons_per_row = spec.neurons_per_row || spec.neuron_count;
		const number_of_rows = Math.ceil(spec.neuron_count / neurons_per_row);

		const x_space_between_neurons = neurons_per_row == 1 ? 0 : spec.width / (neurons_per_row - 1);
		const y_space_between_neurons = number_of_rows == 1 ? 0 : spec.height / (number_of_rows - 1);

		const row_x_start = (-x_space_between_neurons * (neurons_per_row - 1)) / 2;

		const neurons = [];

		let index_in_row = 0;
		let y = layer_start_y;

		for (let i = 0; i < spec.neuron_count; i++) {
			neurons.push(new Neuron(row_x_start + index_in_row * x_space_between_neurons, y));
			index_in_row++;
			if (index_in_row == neurons_per_row) {
				index_in_row = 0;
				y -= y_space_between_neurons;
			}
		}

		this.neurons = neurons;
	}
}

// Alow callers to pass in a filter when requesting the links
export type LinkFilter = (links: Link[]) => Link[];

export const allLinks: LinkFilter = (links) => links;

// XXX: Whats with the 0.1 * length ?
export function makeTopNLinksFilter(n: number) {
	return function (links: Link[]) {
		const length = links.length;
		if (length <= n) {
			return links;
		}
		const sortedLinks = [...links].sort(
			(l1: Link, l2: Link) => Math.abs(l2.weight) - Math.abs(l1.weight)
		);
		return sortedLinks.slice(0, Math.min(n, 0.1 * length));
	};
}

export function neighborsFilter(neuron: Neuron) {
	return function (links: Link[]) {
		return links.filter((link) => link.a == neuron || link.b == neuron);
	};
}

export const applyActivation: LinkFilter = (links) => {
	if (!links.find((link) => link.a.activation)) {
		// A small optimization
		return links;
	}
	return links.map((link) => new Link(link.a, link.b, link.weight * (1 + 0.5 * link.a.activation)));
};

export class DenseNetwork {
	readonly layers: Layer[];
	readonly classes: string[];
	readonly outputLayer: Layer;

	constructor(layer_spacing: number, ...layer_specs: LayerSpec[]) {
		this.layers = [];
		let layer_y = 0;
		for (const spec of layer_specs) {
			this.layers.push(new Layer(layer_y, spec));
			layer_y -= spec.height + layer_spacing;
		}
		this.outputLayer = this.layers[this.layers.length - 1];
		this.classes = this.outputLayer.classes!;
	}

	getLinks(weights: LayerVariable[], activations: number[][], linkFilter: LinkFilter) {
		if (activations) {
			for (const [layer, activationsForLayer] of zip2(this.layers, activations)) {
				for (const [neuron, activation] of zip2(layer.neurons, activationsForLayer)) {
					neuron.activation = activation;
				}
			}
		} else {
			for (const layer of this.layers) {
				for (const neuron of layer.neurons) {
					neuron.activation = 0;
				}
			}
		}

		const links: Link[] = [];
		for (const [from_layer, to_layer, weights_between_layers_tensor] of zip3(
			this.layers.slice(0, -1),
			this.layers.slice(1),
			weights
		)) {
			const weights_between_layers = weights_between_layers_tensor.read().arraySync() as number[][];

			const layerLinks = [];

			for (const [from_neuron, outgoing_weights_for_neuron] of zip2(
				from_layer.neurons,
				weights_between_layers
			)) {
				for (const [to_neuron, weight] of zip2(to_layer.neurons, outgoing_weights_for_neuron)) {
					const link = new Link(from_neuron, to_neuron, weight);
					layerLinks.push(link);
				}
			}
			const filteredLayerLinks = linkFilter(layerLinks);
			links.push(...filteredLayerLinks);
		}

		return links;
	}
}
