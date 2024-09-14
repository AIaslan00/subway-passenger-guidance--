<?php
$servername = "localhost";
$username = "root";
$password = "AI023";
$dbname = "metro";

// Veritabanı bağlantısını oluştur
$conn = new mysqli($servername, $username, $password, $dbname);

// Bağlantıyı kontrol et
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// SQL sorgusu
$sql = "SELECT entering_count, exiting_count, vagon_count FROM metro_data ORDER BY id DESC LIMIT 1";
$result = $conn->query($sql);

$data = array();

if ($result->num_rows > 0) {
    $row = $result->fetch_assoc();
    
    // Verileri int türüne dönüştür
    $data['entering_count'] = intval($row['entering_count']);
    $data['exiting_count'] = intval($row['exiting_count']);
    $data['vagon_count'] = intval($row['vagon_count']);
} 

$conn->close();

// JSON formatında sonuç döndür
header('Content-Type: application/json');
echo json_encode($data);
?>
