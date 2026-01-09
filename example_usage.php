<?php
/**
 * Ví dụ sử dụng MedicalChatbotAPI
 */

require_once 'MedicalChatbotAPI.php';

// Khởi tạo API client
$api = new MedicalChatbotAPI('http://localhost:5000');

// Kiểm tra server
echo "=== Health Check ===\n";
$health = $api->healthCheck();
if ($health) {
    echo "Server Status: " . ($health['status'] ?? 'unknown') . "\n";
    echo "Model Loaded: " . ($health['model_loaded'] ? 'Yes' : 'No') . "\n";
    echo "Device: " . ($health['device'] ?? 'unknown') . "\n";
    echo "Number of Classes: " . ($health['num_classes'] ?? 0) . "\n";
} else {
    echo "Không thể kết nối đến server!\n";
    exit(1);
}

echo "\n=== Single Prediction ===\n";
$question = "Tôi đang cảm thấy mệt mỏi, chóng mặt và nhịp tim không đều. Tôi có thể đang bị bệnh gì?";
$result = $api->predict($question);

if ($result) {
    echo "Question: " . $result['question'] . "\n";
    echo "Is Confident: " . ($result['is_confident'] ? 'Yes' : 'No') . "\n";
    echo "Max Confidence: " . number_format($result['max_confidence'] * 100, 2) . "%\n\n";
    
    echo "Predictions:\n";
    foreach ($result['predictions'] as $i => $pred) {
        echo ($i + 1) . ". " . $pred['disease'] . " - " . 
             number_format($pred['confidence_percent'], 2) . "%\n";
    }
} else {
    echo "Lỗi khi dự đoán!\n";
}

echo "\n=== Batch Prediction ===\n";
$questions = [
    "Tôi đang cảm thấy mệt mỏi, chóng mặt và nhịp tim không đều.",
    "Tôi hay quên mất mình đang làm gì và mục đích của hành động đó.",
    "Tôi đang cảm thấy suy giảm chức năng thận, hội chứng thận hư."
];

$batchResults = $api->predictBatch($questions);

if ($batchResults) {
    foreach ($batchResults as $i => $result) {
        if (isset($result['error'])) {
            echo "Question " . ($i + 1) . ": Error - " . $result['error'] . "\n";
        } else {
            echo "Question " . ($i + 1) . ": " . $result['question'] . "\n";
            echo "  Top prediction: " . $result['predictions'][0]['disease'] . 
                 " (" . number_format($result['predictions'][0]['confidence_percent'], 2) . "%)\n";
        }
    }
} else {
    echo "Lỗi khi dự đoán batch!\n";
}

echo "\n=== Model Info ===\n";
$info = $api->getInfo();
if ($info) {
    echo "Model Path: " . ($info['model_path'] ?? 'unknown') . "\n";
    echo "Device: " . ($info['device'] ?? 'unknown') . "\n";
    echo "Max Length: " . ($info['max_length'] ?? 0) . "\n";
    echo "Number of Classes: " . ($info['num_classes'] ?? 0) . "\n";
}

?>









