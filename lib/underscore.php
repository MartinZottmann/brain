<?php
function underscore_isArray($mixed) {
	return is_array($mixed) && count(array_filter(array_keys($mixed), 'is_string')) === 0;
}

function underscore_pluck(array $list, $propertyName) {
	$output = [];
	foreach ($list as $v) {
		if (array_key_exists($propertyName, $v)) {
			$output[] = $v[$propertyName];
		}
	}
	return $output;
}