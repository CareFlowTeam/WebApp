import os
import json
from typing import Optional
import pandas as pd

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

class PillInfoService:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(base_path, 'data', 'pill_data_final_remake 1.json')
        pkl_path = os.path.join(base_path, 'data', 'pill_data.pkl')
        
        print("🎙️ 서비스 엔진을 초기화합니다...")
        
        # 1. JSON 데이터 로드
        self.pill_data_json = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.pill_data_json = json.load(f)
                print(f"✅ JSON 상세 DB 로드 완료: {len(self.pill_data_json)}개의 항목")
            except Exception as e:
                print(f"⚠️ JSON 로드 실패: {e}")

        # 2. Pandas DB 로드
        self.df = pd.DataFrame()
        if os.path.exists(pkl_path):
            try:
                self.df = pd.read_pickle(pkl_path)
                print(f"✅ pkl DB 로드 완료: {len(self.df)}개의 항목")
            except Exception as e:
                print(f"⚠️ pkl 로드 실패: {e}")

    def _play_tts(self, pill_name):
        """내부 TTS 재생 함수 (에러 방지를 위해 클래스 내부에 정의)"""
        try:
            if pyttsx3 is not None:
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                # 사용자 요청: 완벽한 여성 목소리 설정
                for voice in voices:
                    if 'Korean' in voice.name or 'Heami' in voice.name or 'Arami' in voice.name:
                        engine.setProperty('voice', voice.id)
                        break
                
                engine.setProperty('rate', 170)
                text = f"검색하신 약은 {pill_name}입니다."
                print(f"🔊 음성 출력: {text}")
                engine.say(text)
                engine.runAndWait()
                engine.stop()
        except Exception as e:
            print(f"📢 TTS 출력 실패: {e}")

    def search_and_announce(self, pill_name):
        if not pill_name:
            return None

        res = None
        full_text = str(pill_name).strip()
        # 검색어 후보들: 전체 문장 + 공백으로 나눈 단어들
        query_candidates = [full_text] + full_text.split()

        # --- [1순위] JSON 상세 데이터 검색 ---
        if self.pill_data_json:
            for key, info in self.pill_data_json.items():
                if isinstance(info, list): continue
                db_name = info.get('name', '')
                
                # 후보 단어 중 하나라도 DB의 약 이름에 포함되어 있는지 확인
                for q in query_candidates:
                    if len(q) < 2: continue # 한 글자(정, 알 등)는 무시
                    if q in db_name or db_name in q:
                        res = {
                            "제품명": db_name,
                            "업체명": info.get('manufacturer', '업체 없음'),
                            "효능": info.get('effect', '정보 없음'),
                            "용법": info.get('usage', '정보 없음'),
                            "주의사항": info.get('caution', '정보 없음'),
                            "보관법": info.get('storage', '정보 없음'),
                            "분류": "상세정보 확인됨"
                        }
                        print(f"🎯 JSON 매칭 성공: {res['제품명']}")
                        break
                if res: break

        # --- [2순위] Pandas/PKL 검색 (JSON 실패 시) ---
        if res is None and self.df is not None:
            try:
                # 수정: self.df가 DataFrame인지 확실히 체크 (에러 방지 핵심)
                if hasattr(self.df, 'empty') and not self.df.empty:
                    # columns 속성 에러 방지를 위해 직접 리스트로 가져오기
                    cols = list(self.df.columns)
                    search_col = '품목명' if '품목명' in cols else cols[0]
                    
                    for q in query_candidates:
                        if len(q) < 2: continue
                        mask = self.df[search_col].astype(str).str.contains(q, na=False, case=False)
                        matched_df = self.df[mask]
                        
                        if len(matched_df) > 0:
                            item = matched_df.iloc[0]
                            res = {
                                "제품명": str(item.get('품목명', '이름 없음')),
                                "업체명": str(item.get('업체명', '업체 없음')),
                                "성분": str(item.get('주성분', '성분 없음')),
                                "분류": str(item.get('전문일반구분', '분류 없음'))
                            }
                            print(f"🎯 DB 매칭 성공: {res['제품명']}")
                            break
            except Exception as e:
                print(f"⚠️ 보조 검색 로직 예외(무시 가능): {e}")

        # --- [결과 처리] ---
        if res:
            self._play_tts(res['제품명'])
            return res
        
        print(f"🔍 '{full_text[:20]}...'와(과) 일치하는 약 정보를 찾지 못했습니다.")
        return None