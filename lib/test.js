var net = new brain.NeuralNetwork();
/*
net.train([
	{input: [0, 0, 0, 0], output: [0, 0, 0, 0]},
	{input: [1, 0, 0, 0], output: [1, 0, 0, 0]},
	{input: [0, 1, 0, 0], output: [0, 1, 0, 0]},
	{input: [0, 0, 1, 0], output: [0, 0, 1, 0]},
	{input: [0, 0, 0, 1], output: [0, 0, 0, 1]}
]);

var output = net.run([1, 1, 1, 1]);  // [0.987]
*/
      var net = new brain.NeuralNetwork();

      net.train([{input: {}, output: {}},
                 {input: { y: 1 }, output: { answer: 1 }},
                 {input: { x: 1 }, output: { answer: 1 }},
                 {input: { x: 1, y: 1 }, output: {}}]);


      var output = net.run({x: 1});

console.log(output);
/*console.log(net.test([
	{input: [0, 0, 0, 0], output: [0, 0, 0, 0]},
	{input: [1, 0, 0, 0], output: [1, 0, 0, 0]},
	{input: [0, 1, 0, 0], output: [0, 1, 0, 0]},
	{input: [0, 0, 1, 0], output: [0, 0, 1, 0]},
	{input: [0, 0, 0, 1], output: [0, 0, 0, 1]},
	{input: [1, 1, 0, 0], output: [1, 1, 0, 0]},
	{input: [0, 1, 1, 0], output: [0, 1, 1, 0]},
	{input: [0, 0, 1, 1], output: [0, 0, 1, 1]},
	{input: [1, 1, 1, 0], output: [1, 1, 1, 0]},
	{input: [0, 1, 1, 1], output: [0, 1, 1, 1]},
	{input: [1, 1, 1, 1], output: [1, 1, 1, 1]},
]));*/
console.log(net);