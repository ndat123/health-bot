"""
Module tiền xử lý dữ liệu tiếng Việt cho chẩn đoán bệnh
Xử lý:
- Chuẩn hóa văn bản tiếng Việt
- Loại bỏ dấu (nếu cần)
- Xử lý missing values
- Làm sạch text
"""

import re
import unicodedata
import pandas as pd
from typing import Tuple, Optional


class VietnameseTextPreprocessor:
    """Class xử lý văn bản tiếng Việt"""
    
    def __init__(self, remove_accents: bool = False):
        """
        Khởi tạo preprocessor
        
        Args:
            remove_accents: Có loại bỏ dấu tiếng Việt không (mặc định: False)
        """
        self.remove_accents = remove_accents
        
        # Stopwords tiếng Việt đơn giản
        self.stopwords = {
            'tôi', 'bị', 'đang', 'có', 'thể', 'là', 'của', 'và', 'các',
            'được', 'cho', 'từ', 'với', 'này', 'để', 'trong', 'không',
            'có thể', 'gì', 'cảm thấy', 'hiện', 'hay', 'đã'
        }
    
    def remove_vietnamese_accents(self, text: str) -> str:
        """
        Loại bỏ dấu tiếng Việt
        
        Args:
            text: Văn bản đầu vào
            
        Returns:
            Văn bản không dấu
        """
        if not text:
            return ""
        
        # Normalize unicode
        text = unicodedata.normalize('NFD', text)
        
        # Loại bỏ các ký tự dấu
        text = ''.join(char for char in text 
                      if unicodedata.category(char) != 'Mn')
        
        # Xử lý các ký tự đặc biệt tiếng Việt
        replacements = {
            'đ': 'd', 'Đ': 'D',
            'ð': 'd', 'Ð': 'D'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Làm sạch văn bản
        
        Args:
            text: Văn bản đầu vào
            
        Returns:
            Văn bản đã làm sạch
        """
        if not isinstance(text, str):
            return ""
        
        # Chuyển về lowercase
        text = text.lower()
        
        # Loại bỏ dấu nếu cần
        if self.remove_accents:
            text = self.remove_vietnamese_accents(text)
        
        # Loại bỏ các ký tự đặc biệt, giữ lại chữ cái, số và khoảng trắng
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Loại bỏ stopwords
        
        Args:
            text: Văn bản đầu vào
            
        Returns:
            Văn bản đã loại bỏ stopwords
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return ' '.join(filtered_words)
    
    def preprocess(self, text: str, remove_stopwords: bool = False) -> str:
        """
        Tiền xử lý hoàn chỉnh
        
        Args:
            text: Văn bản đầu vào
            remove_stopwords: Có loại bỏ stopwords không
            
        Returns:
            Văn bản đã được xử lý
        """
        text = self.clean_text(text)
        
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        return text


class DiseaseDataLoader:
    """Class load và xử lý dữ liệu bệnh"""
    
    def __init__(self, csv_path: str, remove_accents: bool = False):
        """
        Khởi tạo data loader
        
        Args:
            csv_path: Đường dẫn đến file CSV
            remove_accents: Có loại bỏ dấu không
        """
        self.csv_path = csv_path
        self.preprocessor = VietnameseTextPreprocessor(remove_accents)
        self.df = None
        self.disease_mapping = {}
        self.reverse_mapping = {}
    
    def load_data(self) -> pd.DataFrame:
        """
        Load dữ liệu từ CSV
        
        Returns:
            DataFrame đã xử lý
        """
        print(f"📂 Đang đọc dữ liệu từ: {self.csv_path}")
        
        # Đọc CSV
        self.df = pd.read_csv(self.csv_path)
        
        print(f"✓ Đã đọc {len(self.df)} dòng dữ liệu")
        
        # Kiểm tra columns
        required_columns = ['Disease', 'Question']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Thiếu cột '{col}' trong dataset!")
        
        return self.df
    
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Xử lý giá trị bị thiếu
        
        Returns:
            DataFrame đã xử lý
        """
        print("\n🔍 Kiểm tra dữ liệu thiếu...")
        
        # Đếm giá trị thiếu
        missing_counts = self.df.isnull().sum()
        
        if missing_counts.sum() > 0:
            print(f"⚠️  Tìm thấy {missing_counts.sum()} giá trị thiếu:")
            for col, count in missing_counts.items():
                if count > 0:
                    print(f"   - {col}: {count} dòng")
            
            # Loại bỏ các dòng có giá trị thiếu
            before = len(self.df)
            self.df = self.df.dropna()
            after = len(self.df)
            print(f"✓ Đã loại bỏ {before - after} dòng có dữ liệu thiếu")
        else:
            print("✓ Không có dữ liệu thiếu")
        
        return self.df
    
    def preprocess_text(self, remove_stopwords: bool = False) -> pd.DataFrame:
        """
        Tiền xử lý văn bản
        
        Args:
            remove_stopwords: Có loại bỏ stopwords không
            
        Returns:
            DataFrame đã xử lý
        """
        print("\n🔧 Đang tiền xử lý văn bản...")
        
        # Xử lý cột Question
        self.df['Question_Processed'] = self.df['Question'].apply(
            lambda x: self.preprocessor.preprocess(x, remove_stopwords)
        )
        
        # Xử lý cột Disease (để đồng nhất)
        self.df['Disease_Processed'] = self.df['Disease'].apply(
            lambda x: str(x).strip()
        )
        
        print("✓ Hoàn thành tiền xử lý văn bản")
        
        return self.df
    
    def create_disease_mapping(self) -> Tuple[dict, dict]:
        """
        Tạo mapping giữa tên bệnh và ID số
        
        Returns:
            Tuple (disease_to_id, id_to_disease)
        """
        print("\n🗂️  Tạo mapping bệnh...")
        
        # Lấy danh sách các bệnh duy nhất
        unique_diseases = sorted(self.df['Disease_Processed'].unique())
        
        # Tạo mapping
        self.disease_mapping = {disease: idx for idx, disease in enumerate(unique_diseases)}
        self.reverse_mapping = {idx: disease for disease, idx in self.disease_mapping.items()}
        
        # Thêm cột label (ID số)
        self.df['label'] = self.df['Disease_Processed'].map(self.disease_mapping)
        
        print(f"✓ Đã tạo mapping cho {len(unique_diseases)} loại bệnh")
        
        return self.disease_mapping, self.reverse_mapping
    
    def get_statistics(self) -> dict:
        """
        Lấy thống kê về dataset
        
        Returns:
            Dictionary chứa thống kê
        """
        stats = {
            'total_samples': len(self.df),
            'num_diseases': len(self.disease_mapping),
            'samples_per_disease': self.df['label'].value_counts().to_dict(),
            'avg_question_length': self.df['Question_Processed'].apply(len).mean(),
            'min_samples': self.df['label'].value_counts().min(),
            'max_samples': self.df['label'].value_counts().max(),
        }
        
        return stats
    
    def print_statistics(self):
        """In thống kê dataset"""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("📊 THỐNG KÊ DATASET")
        print("="*70)
        print(f"Tổng số mẫu: {stats['total_samples']:,}")
        print(f"Số loại bệnh: {stats['num_diseases']}")
        print(f"Độ dài câu hỏi trung bình: {stats['avg_question_length']:.1f} ký tự")
        print(f"Số mẫu ít nhất cho 1 bệnh: {stats['min_samples']}")
        print(f"Số mẫu nhiều nhất cho 1 bệnh: {stats['max_samples']}")
        
        # Top 5 bệnh có nhiều mẫu nhất
        top_diseases = sorted(stats['samples_per_disease'].items(), 
                            key=lambda x: x[1], reverse=True)[:5]
        print(f"\n📈 Top 5 bệnh có nhiều mẫu nhất:")
        for label_id, count in top_diseases:
            disease_name = self.reverse_mapping[label_id]
            print(f"   {disease_name}: {count} mẫu")
        
        print("="*70)
    
    def prepare_data(self, remove_stopwords: bool = False) -> pd.DataFrame:
        """
        Pipeline xử lý dữ liệu hoàn chỉnh
        
        Args:
            remove_stopwords: Có loại bỏ stopwords không
            
        Returns:
            DataFrame đã xử lý hoàn chỉnh
        """
        print("\n🚀 BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU")
        print("="*70)
        
        # Load data
        self.load_data()
        
        # Xử lý missing values
        self.handle_missing_values()
        
        # Tiền xử lý văn bản
        self.preprocess_text(remove_stopwords)
        
        # Tạo mapping
        self.create_disease_mapping()
        
        # In thống kê
        self.print_statistics()
        
        return self.df


# Test module
if __name__ == "__main__":
    # Test preprocessor
    print("Test Vietnamese Text Preprocessor:")
    print("="*70)
    
    preprocessor = VietnameseTextPreprocessor(remove_accents=False)
    
    test_texts = [
        "Tôi đang cảm thấy đau đầu, sốt cao và mệt mỏi.",
        "Tôi hay bị buồn nôn, chóng mặt và khó thở.",
        "Tôi hiện đang có các triệu chứng như ho, sổ mũi và đau họng."
    ]
    
    for text in test_texts:
        cleaned = preprocessor.preprocess(text, remove_stopwords=False)
        print(f"Gốc: {text}")
        print(f"Xử lý: {cleaned}")
        print()
    
    # Test data loader (nếu có file CSV)
    import os
    if os.path.exists("ViMedical_Disease.csv"):
        print("\n" + "="*70)
        print("Test Data Loader:")
        print("="*70)
        
        loader = DiseaseDataLoader("ViMedical_Disease.csv", remove_accents=False)
        df = loader.prepare_data(remove_stopwords=False)
        
        print(f"\n✓ Dataset đã sẵn sàng với {len(df)} mẫu!")
        print(f"✓ Columns: {list(df.columns)}")
        print(f"\nMẫu dữ liệu đầu tiên:")
        print(df[['Disease_Processed', 'Question_Processed', 'label']].head(1))


