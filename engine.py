from src.analyzer import MedicineAnalyzer

class SearchEngine:
    """
    이미지 분석 + RAG 검색을 통합하는 엔진
    """
    
    def __init__(self, db, rag_engine):
        self.analyzer = MedicineAnalyzer()
        self.db = db
        self.rag_engine = rag_engine

    def run_process(self, image_path):
        """
        전체 프로세스 실행:
        1. 이미지에서 OCR로 텍스트 추출
        2. 이미지 특징(색상, 모양) 추출
        3. RAG로 정보 검색 및 응답 생성
        """
        print(f"[처리 시작] {image_path}")
        
        # 1. AI 분석기로 특징 추출
        features = self.analyzer.analyze(image_path)
        
        # 2. OCR로 추출한 텍스트를 쿼리로 사용
        ocr_text = features.get('text', '')
        
        if not ocr_text:
            return {
                "error": "알약에서 텍스트를 인식할 수 없습니다.",
                "features": features,
                "suggestion": "이미지를 더 선명하게 촬영해주세요."
            }
        
        print(f"[OCR 결과] {ocr_text}")
        
        # 3. RAG로 정보 검색 및 생성
        result = self.rag_engine.generate_response(
            query=ocr_text,
            ocr_text=ocr_text
        )
        
        # 4. 이미지 특징 정보 추가
        result['image_features'] = {
            'color': features.get('color'),
            'shape': features.get('shape'),
            'text': features.get('text')
        }
        
        print(f"[처리 완료] {ocr_text}")
        
        return result
