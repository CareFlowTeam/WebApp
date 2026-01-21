import json
import os
from typing import List, Dict
from difflib import SequenceMatcher

class PillDatabase:
    """
    의약품 데이터베이스 관리
    - JSON 파일 기반 저장
    - 검색 및 유사도 매칭
    """
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.data = self._load_database()
    
    def _load_database(self) -> Dict:
        """
        데이터베이스 파일을 로드합니다.
        파일이 없으면 샘플 데이터를 생성합니다.
        """
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"데이터베이스 로드 오류: {e}")
                return self._create_sample_data()
        else:
            # 샘플 데이터 생성
            sample_data = self._create_sample_data()
            self._save_database(sample_data)
            return sample_data
    
    def _create_sample_data(self) -> Dict:
        """
        샘플 의약품 데이터 생성
        """
        return {
            "K25": {
                "name": "케이25정 (아세트아미노펜)",
                "effect": "해열, 진통제. 두통, 발열, 근육통 등을 완화합니다.",
                "caution": "과다 복용 시 간 손상 위험. 알코올과 함께 복용 금지.",
                "ingredients": "아세트아미노펜 500mg",
                "dosage": "성인 1회 1~2정, 1일 3~4회"
            },
            "타이레놀": {
                "name": "타이레놀정 500mg",
                "effect": "두통, 치통, 월경통, 근육통, 신경통 등의 진통 및 해열",
                "caution": "간 질환 환자 주의. 1일 최대 4000mg 초과 금지.",
                "ingredients": "아세트아미노펜 500mg",
                "dosage": "성인 1회 1~2정, 4~6시간마다 복용"
            },
            "아스피린": {
                "name": "아스피린정 100mg",
                "effect": "혈전 예방, 해열, 진통, 항염증 효과",
                "caution": "위장 장애 가능. 출혈 위험 증가. 임산부 금지.",
                "ingredients": "아세틸살리실산 100mg",
                "dosage": "성인 1회 100~300mg, 1일 1회"
            },
            "이부프로펜": {
                "name": "부루펜정 200mg",
                "effect": "소염, 진통, 해열. 관절염, 두통, 생리통 등에 효과",
                "caution": "위장 장애, 신장 기능 저하 가능. 장기 복용 주의.",
                "ingredients": "이부프로펜 200mg",
                "dosage": "성인 1회 200~400mg, 1일 3~4회"
            }
        }
    
    def _save_database(self, data: Dict):
        """
        데이터베이스를 파일로 저장합니다.
        """
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_medicine_info(self, pill_name: str) -> Dict:
        """
        정확한 약품명으로 정보를 검색합니다.
        """
        # 대소문자 무시하고 검색
        pill_name_lower = pill_name.lower()
        
        for key, value in self.data.items():
            if pill_name_lower in key.lower() or pill_name_lower in value.get('name', '').lower():
                return {
                    "found": True,
                    "name": value['name'],
                    "effect": value['effect'],
                    "caution": value['caution'],
                    "ingredients": value.get('ingredients', '정보 없음'),
                    "dosage": value.get('dosage', '정보 없음')
                }
        
        return {
            "found": False,
            "name": pill_name,
            "message": "데이터베이스에서 해당 약품을 찾을 수 없습니다."
        }
    
    def search_similar(self, query: str, threshold: float = 0.4) -> List[Dict]:
        """
        유사한 약품명을 검색합니다.
        """
        results = []
        query_lower = query.lower()
        
        for key, value in self.data.items():
            # 유사도 계산
            key_similarity = SequenceMatcher(None, query_lower, key.lower()).ratio()
            name_similarity = SequenceMatcher(None, query_lower, value.get('name', '').lower()).ratio()
            
            max_similarity = max(key_similarity, name_similarity)
            
            if max_similarity >= threshold:
                results.append({
                    "name": value['name'],
                    "effect": value['effect'],
                    "caution": value['caution'],
                    "ingredients": value.get('ingredients', '정보 없음'),
                    "dosage": value.get('dosage', '정보 없음'),
                    "similarity": max_similarity
                })
        
        # 유사도 순으로 정렬
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results
    
    def add_medicine(self, key: str, info: Dict):
        """
        새로운 약품 정보를 추가합니다.
        """
        self.data[key] = info
        self._save_database(self.data)
        print(f"[DB] {key} 추가 완료")
    
    def update_medicine(self, key: str, info: Dict):
        """
        기존 약품 정보를 업데이트합니다.
        """
        if key in self.data:
            self.data[key].update(info)
            self._save_database(self.data)
            print(f"[DB] {key} 업데이트 완료")
        else:
            print(f"[DB] {key}를 찾을 수 없습니다.")
    
    def delete_medicine(self, key: str):
        """
        약품 정보를 삭제합니다.
        """
        if key in self.data:
            del self.data[key]
            self._save_database(self.data)
            print(f"[DB] {key} 삭제 완료")
        else:
            print(f"[DB] {key}를 찾을 수 없습니다.")
