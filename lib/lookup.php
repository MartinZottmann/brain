<?php
include_once 'underscore.php';

/* Functions for turning sparse hashes into arrays and vice versa */

// [{a: 1}, {b: 6, c: 7}] -> {a: 0, b: 1, c: 2}
function lookup_buildLookup($hashes) {
	$hash = array_reduce(
		$hashes,
		function($memo, $hash) {
			return $memo + $hash;
		},
		[]
	);
	return lookup_lookupFromHash($hash);
}

// {a: 6, b: 7} -> {a: 0, b: 1}
function lookup_lookupFromHash($hash) {
	$lookup = [];
	$index = 0;
	foreach ($hash as $k => $v) {
		$lookup[$k] = $index++;
	}
	return $lookup;
}

// {a: 0, b: 1}, {a: 6} -> [6, 0]
function lookup_toArray($lookup, $hash) {
	$array = [];
	foreach ($lookup as $k => $v) {
		$array[$v] = array_key_exists($k, $hash) ? $hash[$k] : 0;
	}
	return $array;
}

// {a: 0, b: 1}, [6, 7] -> {a: 6, b: 7}
function lookup_toHash($lookup, $array) {
	$hash = [];
	foreach ($lookup as $k => $v) {
		$hash[$k] = $array[$v];
	}
	return $hash;
}