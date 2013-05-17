<?php
namespace MartinZottmann\brain;

include_once 'brain.php';

$net = new NeuralNetwork();
print_r($net->train(
	[
/*		['input' => [0, 0, 0, 0], 'output' => [0, 0, 0, 0]],
		['input' => [1, 0, 0, 0], 'output' => [1, 0, 0, 0]],
		['input' => [0, 1, 0, 0], 'output' => [0, 1, 0, 0]],
		['input' => [0, 0, 1, 0], 'output' => [0, 0, 1, 0]],
		['input' => [0, 0, 0, 1], 'output' => [0, 0, 0, 1]],
		['input' => [1, 1, 0, 0], 'output' => [1, 1, 0, 0]],
		['input' => [0, 1, 1, 0], 'output' => [0, 1, 1, 0]],
		['input' => [0, 0, 1, 1], 'output' => [0, 0, 1, 1]],
		['input' => [0, 1, 1, 1], 'output' => [0, 1, 1, 1]],
		['input' => [1, 1, 1, 0], 'output' => [1, 1, 1, 0]],
		['input' => [1, 1, 1, 1], 'output' => [1, 1, 1, 1]],
*/
		['input' => [0, 0], 'output' => [0, 0]],
		['input' => [0, 1], 'output' => [0, 1]],
		['input' => [1, 0], 'output' => [0, 1]],
		['input' => [1, 1], 'output' => [1, 1]]
	]
));

print_r($net->run([0, 1]));
//print_r($net->test());
//print_r($net);