<?php
namespace MartinZottmann\brain;

include_once 'underscore.php';
include_once 'lookup.php';

class NeuralNetwork {
	protected $inputLookup;

	protected $outputLookup;

	protected $learningRate = 0.3;

	protected $momentum = 0.1;

	protected $hiddenSizes = null;

	protected $binaryThresh = 0.5;

	protected $sizes;

	protected $outputLayer;

	// weights for bias nodes
	protected $biases; 

	protected $weights;

	protected $outputs;

	// state for training
	protected $deltas;

	// for momentum
	protected $changes;

	protected $errors;

	public function __construct(array $options = []) {
		if (array_key_exists('learningRate', $options)) {
			$this->learningRate = $options['learningRate'];
		}
		if (array_key_exists('momentum', $options)) {
			$this->momentum = $options['momentum'];
		}
		if (array_key_exists('hiddenLayers', $options)) {
			$this->hiddenSizes = $options['hiddenLayers'];
		}
		if (array_key_exists('binaryThresh', $options)) {
			$this->binaryThresh = $options['binaryThresh'];
		}
	}

	public function initialize(array $sizes) {
		$this->sizes = $sizes;
		$this->outputLayer = count($this->sizes) - 1;
		$this->biases = [];
		$this->weights = [];
		$this->outputs = [];
		$this->deltas = [];
		$this->changes = [];
		$this->errors = [];

		for ($layer = 0; $layer <= $this->outputLayer; ++$layer) {
			$size = $this->sizes[$layer];
			$this->deltas[$layer] = $this->zeros($size);
			$this->errors[$layer] = $this->zeros($size);
			$this->outputs[$layer] = $this->zeros($size);

			if ($layer > 0) {
				$this->biases[$layer] = $this->randos($size);
				$this->weights[$layer] = [];
				$this->changes[$layer] = [];

				for ($node = 0; $node < $size; ++$node) {
					$prevSize = $this->sizes[$layer - 1];
					$this->weights[$layer][$node] = $this->randos($prevSize);
					$this->changes[$layer][$node] = $this->zeros($prevSize);
				}
			}
		}
	}

	public function run($input) {
		if ($this->inputLookup !== null) {
			$input = lookup_toArray($this->inputLookup, $input);
		}

		$output = $this->runInput($input);

		if ($this->outputLookup !== null) {
			$output = lookup_toHash($this->outputLookup, $output);
		}
		return $output;
	}

	public function runInput(array $input) {
		$this->outputs[0] = $input;	// set output state of input layer

		for ($layer = 1; $layer <= $this->outputLayer; ++$layer) {
			for ($node = 0; $node < $this->sizes[$layer]; ++$node) {
				$weights = $this->weights[$layer][$node];

				$sum = $this->biases[$layer][$node];
				for ($k = 0; $k < count($weights); ++$k) {
					$sum += $weights[$k] * $input[$k];
				}
				$this->outputs[$layer][$node] = 1 / (1 + exp(-$sum));
			}
			$output = $input = $this->outputs[$layer];
		}
		return $output;
	}

	public function train(array $data, array $options = []) {
		$data = $this->formatData($data);

		$options += [
			'iterations' => 20000,
			'errorThresh' => 0.005,
			'log' => false,
			'logPeriod' => 10,
			'callback' => null,
			'callbackPeriod' => 10
		];
		extract($options);

		$inputSize = count($data[0]['input']);
		$outputSize = count($data[0]['output']);

		$hiddenSizes = $this->hiddenSizes;
		if ($hiddenSizes === null) {
			$hiddenSizes = [max(3, floor($inputSize / 2))];
		}
		$this->initialize(array_merge([$inputSize], $hiddenSizes, [$outputSize]));

		$error = 1;
		for ($i = 0; $i < $iterations && $error > $errorThresh; ++$i) {
			$sum = 0;
			for ($j = 0; $j < count($data); ++$j) {
				$sum += $this->trainPattern($data[$j]['input'], $data[$j]['output']);
			}
			$error = $sum / count($data);

			if ($log && ($i % $logPeriod === 0)) {
				echo 'iterations: ', $i, ' training error: ', $error, "\n";
			}
			if ($callback && ($i % $callbackPeriod === 0)) {
				$callback(['error' => $error, 'iterations' => $i]);
			}
		}

		return ['error' => $error, 'iterations' => $i];
	}

	public function trainPattern($input, $target) {
		// forward propogate
		$this->runInput($input);

		// back propogate
		$this->calculateDeltas($target);
		$this->adjustWeights();

		$error = $this->mse($this->errors[$this->outputLayer]);
		return $error;
	}

	public function calculateDeltas($target) {
		for ($layer = $this->outputLayer; $layer >= 0; --$layer) {
			for ($node = 0; $node < $this->sizes[$layer]; ++$node) {
				$output = $this->outputs[$layer][$node];

				$error = 0;
				if ($layer == $this->outputLayer) {
					$error = $target[$node] - $output;
				} else {
					$deltas = $this->deltas[$layer + 1];
					for ($k = 0; $k < count($deltas); ++$k) {
						$error += $deltas[$k] * $this->weights[$layer + 1][$k][$node];
					}
				}
				$this->errors[$layer][$node] = $error;
				$this->deltas[$layer][$node] = $error * $output * (1 - $output);
			}
		}
	}

	public function adjustWeights() {
		for ($layer = 1; $layer <= $this->outputLayer; ++$layer) {
			$incoming = $this->outputs[$layer - 1];

			for ($node = 0; $node < $this->sizes[$layer]; ++$node) {
				$delta = $this->deltas[$layer][$node];

				for ($k = 0; $k < count($incoming); ++$k) {
					$change = $this->changes[$layer][$node][$k];

					$change = ($this->learningRate * $delta * $incoming[$k]) + ($this->momentum * $change);

					$this->changes[$layer][$node][$k] = $change;
					$this->weights[$layer][$node][$k] += $change;
				}
				$this->biases[$layer][$node] += $this->learningRate * $delta;
			}
		}
	}

	public function formatData(array $data) {
		// turn sparse hash input into arrays with 0s as filler
		foreach ($data as $datum) {
			if (count($datum['input']) !== 0) {
				$datum = $datum['input'];
				break;
			}
		}
		if (!underscore_isArray($datum)) {
			if ($this->inputLookup === null) {
				$this->inputLookup = lookup_buildLookup(underscore_pluck($data, 'input'));
			}
			$inputLookup = $this->inputLookup;
			$data = array_map(
				function($datum) use ($inputLookup) {
					return array_merge($datum, ['input' => lookup_toArray($inputLookup, $datum['input'])]);
				},
				$data
			);
		}

		foreach ($data as $datum) {
			if (count($datum['output']) !== 0) {
				$datum = $datum['output'];
				break;
			}
		}
		if (!underscore_isArray($datum)) {
			if ($this->outputLookup === null) {
				$this->outputLookup = lookup_buildLookup(underscore_pluck($data, 'output'));
			}
			$outputLookup = $this->outputLookup;
			$data = array_map(
				function($datum) use ($outputLookup) {
					return array_merge($datum, ['output' => lookup_toArray($outputLookup, $datum['output'])]);
				},
				$data
			);
		}

		return $data;
	}

	public function test($data) {
		$data = $this->formatData($data);

		// for binary classification problems with one output node
		$isBinary = count($data[0]['output']) === 1;
		$falsePos = 0;
		$falseNeg = 0;
		$truePos = 0;
		$trueNeg = 0;

		// for classification problems
		$misclasses = [];

		// run each pattern through the trained network and collect
		// error and misclassification statistics
		$sum = 0;
		for ($i = 0; $i < count($data); ++$i) {
			$output = $this->runInput($data[$i]['input']);
			$target = $data[$i]['output'];

			$actual;
			$expected;
			if ($isBinary) {
				$actual = $output[0] > $this->binaryThresh ? 1 : 0;
				$expected = $target[0];
			} else {
				$actual = array_keys($output, max($output))[0];
				$expected = array_keys($target, max($target))[0];
			}

			if ($actual !== $expected) {
				$misclass = $data[$i] + compact('actual', 'expected');
				$misclasses[] = $misclass;
			}

			if ($isBinary) {
				if ($actual === 0 && $expected === 0) {
					$trueNeg++;
				} else if ($actual == 1 && $expected == 1) {
					$truePos++;
				} else if ($actual == 0 && $expected == 1) {
					$falseNeg++;
				} else if ($actual == 1 && $expected == 0) {
					$falsePos++;
				}
			}

			foreach ($output as $k => $v) {
				$errors[] = $target[$k] - $v;
			}
/*			$errors = array_map(
				function($i, $value) use ($target) {
					return $target[$i] - $value;
				},
				$output
			);*/
			$sum += $this->mse($errors);
		}
		$error = $sum / count($data);

		$stats = compact('error', 'misclasses');

		if ($isBinary) {
			$stats = $stats
				+ compact('trueNeg', 'truePos', 'falseNeg', 'falsePos')
				+ [
					'total' => count($data),
					'precision' => $truePos / ($truePos + $falsePos),
					'recall' => $truePos / ($truePos + $falseNeg),
					'accuracy' => ($trueNeg + $truePos) / count($data)
				];
		}
		return $stats;
	}

		/* make json look like:
			{
				layers: [
					{ x: {},
						y: {}},
					{'0': {bias: -0.98771313, weights: {x: 0.8374838, y: 1.245858},
					 '1': {bias: 3.48192004, weights: {x: 1.7825821, y: -2.67899}}},
					{ f: {bias: 0.27205739, weights: {'0': 1.3161821, '1': 2.00436}}}
				]
			}
		*/
		/*
	toJSON: function() {
		$layers = [];
		for ($layer = 0; layer <= $this->outputLayer; layer++) {
			layers[layer] = {};

			$nodes;
			// turn any internal arrays back into hashes for readable json
			if (layer == 0 && $this->inputLookup) {
				nodes = _($this->inputLookup).keys();
			}
			else if (layer == $this->outputLayer && $this->outputLookup) {
				nodes = _($this->outputLookup).keys();
			}
			else {
				nodes = _.range(0, $this->sizes[layer]);
			}

			for ($j = 0; j < nodes.length; j++) {
				$node = nodes[j];
				layers[layer][node] = {};

				if (layer > 0) {
					layers[layer][node].bias = $this->biases[layer][j];
					layers[layer][node].weights = {};
					for ($k in layers[layer - 1]) {
						$index = k;
						if (layer == 1 && $this->inputLookup) {
							index = $this->inputLookup[k];
						}
						layers[layer][node].weights[k] = $this->weights[layer][j][index];
					}
				}
			}
		}
		return { layers: layers, outputLookup:!!$this->outputLookup, inputLookup:!!$this->inputLookup };
	},

	fromJSON: function(json) {
		$size = json.layers.length;
		$this->outputLayer = size - 1;

		$this->sizes = new Array(size);
		$this->weights = new Array(size);
		$this->biases = new Array(size);
		$this->outputs = new Array(size);

		for ($i = 0; i <= $this->outputLayer; i++) {
			$layer = json.layers[i];
			if (i == 0 && (!layer[0] || json.inputLookup)) {
				$this->inputLookup = lookup.lookupFromHash(layer);
			}
			else if (i == $this->outputLayer && (!layer[0] || json.outputLookup)) {
				$this->outputLookup = lookup.lookupFromHash(layer);
			}

			$nodes = _(layer).keys();
			$this->sizes[i] = nodes.length;
			$this->weights[i] = [];
			$this->biases[i] = [];
			$this->outputs[i] = [];

			for ($j in nodes) {
				$node = nodes[j];
				$this->biases[i][j] = layer[node].bias;
				$this->weights[i][j] = _(layer[node].weights).toArray();
			}
		}
		return this;
	},

	 toFunction: function() {
		$json = $this->toJSON();
		// return standalone function that mimics run()
		return new Function("input",
'	$net = ' + JSON.stringify(json) + ';\n\n\
	for ($i = 1; i < net.layers.length; i++) {\n\
		$layer = net.layers[i];\n\
		$output = {};\n\
		\n\
		for ($id in layer) {\n\
			$node = layer[id];\n\
			$sum = node.bias;\n\
			\n\
			for ($iid in node.weights) {\n\
				sum += node.weights[iid] * input[iid];\n\
			}\n\
			output[id] = (1 / (1 + Math.exp(-sum)));\n\
		}\n\
		input = output;\n\
	}\n\
	return output;');
	}
*/
	public function randomWeight() {
		return mt_rand() / mt_getrandmax() * 0.4 - 0.2;
	}

	public function randos($size) {
		$array = [];
		for ($i = 0; $i < $size; ++$i) {
			$array[$i] = $this->randomWeight();
		}
		return $array;
	}

	public function zeros($size) {
		return array_fill(0, $size, 0);
	}

	public function mse($errors) {
		// mean squared error
		$sum = 0;
		for ($i = 0; $i < count($errors); ++$i) {
			$sum += pow($errors[$i], 2);
		}
		return $sum / count($errors);
	}
}