<?php
    error_reporting(E_ALL);
    $f = fopen('file.json', 'w+');

    $data = $_POST['something'];
    
    //$cleanData = json_decode($data);
    //fwrite($f, $cleanData);

    fwrite($f, $data);
    fclose($f);
?>