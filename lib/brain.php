<?php
namespace MartinZottmann\brain;

error_reporting(-1);
set_error_handler(function() { $_ = func_get_args(); throw new \Exception(print_r($_, true)); });
set_time_limit(0);

include_once __DIR__ . DIRECTORY_SEPARATOR . 'neuralnetwork.php';