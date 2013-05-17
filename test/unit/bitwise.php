<?php
namespace MartinZottmann\brain;

include_once __DIR__ . '/../../lib/brain.php';

function testBitwise($data, $op) {
	$wiggle = 0.1;
	$net = new NeuralNetwork();
	$net->train($data, ['errorThresh' => 0.003]);

	foreach($data as $k => $v) {
		$output = $net->run($v['input']);
		$target = $v['output'];
		foreach($target as $k1 => $v1) {
			assert('$output[$k1] < ($target[$k1] + $wiggle) && $output[$k1] > ($target[$k1] - $wiggle) /*
				failed to train ' . $op . ' - output: ' . print_r($output[$k1], true) . ' target: ' . print_r($target[$k1], true) . '
			*/');
		}
	}
}

//describe('bitwise functions', function() {

//	it('NOT function', function() {
		$not = [
			['input' => [0], 'output' => [1]],
			['input' => [1], 'output' => [0]]
		];
		testBitwise($not, 'not');
//	})

//	it('XOR function', function() {
		$xor = [
			['input' => [0, 0], 'output' => [0]],
			['input' => [1, 0], 'output' => [1]],
			['input' => [0, 1], 'output' => [1]],
			['input' => [1, 1], 'output' => [0]]
		];
		testBitwise($xor, 'xor');
//	})

//	it('OR function', function() {
		$or = [
			['input' => [0, 0], 'output' => [0]],
			['input' => [1, 0], 'output' => [1]],
			['input' => [0, 1], 'output' => [1]],
			['input' => [1, 1], 'output' => [1]]
		];
		testBitwise($or, 'or');
//	});

//	it('AND function', function() {
		$and = [
			['input' => [0, 0], 'output' => [0]],
			['input' => [1, 0], 'output' => [0]],
			['input' => [0, 1], 'output' => [0]],
			['input' => [1, 1], 'output' => [1]]
		];
		testBitwise($and, 'and');
//	})
//})
