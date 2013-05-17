<?php
namespace MartinZottmann\brain;

include_once __DIR__ . '/../../lib/brain.php';

//describe('hash input and output', function() {
//	it('runs correctly with array input and output', function() {
		$net = new NeuralNetwork();

		$net->train([
			['input' => [0, 0], 'output' => [0]],
			['input' => [0, 1], 'output' => [1]],
			['input' => [1, 0], 'output' => [1]],
			['input' => [1, 1], 'output' => [0]]
		]);
		$output = $net->run([1, 0]);

		assert('$output[0] > 0.9 /* output: ' . $output[0] . ' */');
//	})

//	it('runs correctly with hash input', function() {
		$net = new NeuralNetwork();

		$net->train([
			['input' => ['x' => 0, 'y' => 0], 'output' => [0]],
			['input' => ['x' => 0, 'y' => 1], 'output' => [1]],
			['input' => ['x' => 1, 'y' => 0], 'output' => [1]],
			['input' => ['x' => 1, 'y' => 1], 'output' => [0]]
		]);
		$output = $net->run(['x' => 1, 'y' => 0]);

		assert('$output[0] > 0.9 /* output: ' . $output[0] . ' */');
//	})

//	it('runs correctly with hash output', function() {
		$net = new NeuralNetwork();

		$net->train([
			['input' => [0, 0], 'output' => ['answer' => 0]],
			['input' => [0, 1], 'output' => ['answer' => 1]],
			['input' => [1, 0], 'output' => ['answer' => 1]],
			['input' => [1, 1], 'output' => ['answer' => 1]]
		]);
		$output = $net->run([1, 0]);

		assert('$output[\'answer\'] > 0.9 /* output: ' . $output['answer'] . ' */');
//	})

//	it('runs correctly with hash input and output', function() {
		$net = new NeuralNetwork(['learningRate' => 0.2]);

		$net->train([
			['input' => ['x' => 0, 'y' => 0], 'output' => ['answer' => 0]],
			['input' => ['x' => 0, 'y' => 1], 'output' => ['answer' => 1]],
			['input' => ['x' => 1, 'y' => 0], 'output' => ['answer' => 1]],
			['input' => ['x' => 1, 'y' => 1], 'output' => ['answer' => 0]]
		]);
		$output = $net->run(['x' => 1, 'y' => 0]);

		assert('$output[\'answer\'] > 0.9 /* output: ' . $output['answer'] . ' */');
//	})

//	it('runs correctly with sparse hashes', function() {
		$net = new NeuralNetwork();

		$net->train([
			['input' => [], 'output' => []],
			['input' => ['y' => 1], 'output' => ['answer' => 1]],
			['input' => ['x' => 1], 'output' => ['answer' => 1]],
			['input' => ['x' => 1, 'y' => 1], 'output' => []]
		]);
		$output = $net->run(['x' => 1]);

		assert('$output[\'answer\'] > 0.9 /* output: ' . $output['answer'] . ' */');
//	})

//	it('runs correctly with unseen input', function() {
		$net = new NeuralNetwork();

		$net->train([
			['input' => [], 'output' => []],
			['input' => ['y' => 1], 'output' => ['answer' => 1]],
			['input' => ['x' => 1], 'output' => ['answer' => 1]],
			['input' => ['x' => 1, 'y' => 1], 'output' => []]
		]);
		$output = $net->run(['x' => 1, 'z' => 1]);

		assert('$output[\'answer\'] > 0.9 /* output: ' . $output['answer'] . ' */');
//	})
//})
