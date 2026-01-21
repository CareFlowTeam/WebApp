import anthropic
import os
from typing import List, Dict

class RAGEngine:
    """
    검색증강생성(RAG) 엔진
    - 데이터베이스에서 관련 정보 검색
    - Claude API를 사용하여 자연어 응답 생성
    """
    
    def __init__(self, database):
        self.db = database
        # Claude API 키 설정 (환경변수에서 가져오기)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = None
            print("경고: ANTHROPIC_API_KEY가 설정되지 않았습니다. RAG 기능이 제한됩니다.")
    
    def retrieve_context(self, query: str) -> List[Dict]:
        """
        데이터베이스에서 관련 정보를 검색합니다.
        """
        # 1. 정확한 매칭 시도
        exact_match = self.db.get_medicine_info(query)
        
        # 2. 유사한 약품명 검색
        similar_medicines = self.db.search_similar(query)
        
        # 3. 컨텍스트 구성
        contexts = []
        
        if exact_match and exact_match.get("found"):
            contexts.append(exact_match)
        
        contexts.extend(similar_medicines[:3])  # 상위 3개만 사용
        
        return contexts
    
    def format_context(self, contexts: List[Dict]) -> str:
        """
        검색된 정보를 프롬프트에 포함할 수 있는 형태로 포맷팅합니다.
        """
        if not contexts:
            return "검색된 정보가 없습니다."
        
        formatted = "검색된 의약품 정보:\n\n"
        
        for i, ctx in enumerate(contexts, 1):
            formatted += f"[정보 {i}]\n"
            formatted += f"약품명: {ctx.get('name', '알 수 없음')}\n"
            formatted += f"효능: {ctx.get('effect', '정보 없음')}\n"
            formatted += f"주의사항: {ctx.get('caution', '정보 없음')}\n"
            
            if 'ingredients' in ctx:
                formatted += f"성분: {ctx.get('ingredients', '정보 없음')}\n"
            
            if 'dosage' in ctx:
                formatted += f"용법용량: {ctx.get('dosage', '정보 없음')}\n"
            
            formatted += "\n"
        
        return formatted
    
    def generate_response(self, query: str, ocr_text: str = None) -> Dict:
        """
        RAG를 사용하여 최종 응답을 생성합니다.
        """
        # 1. 관련 정보 검색
        contexts = self.retrieve_context(query)
        
        # Claude API를 사용할 수 없는 경우 검색 결과만 반환
        if not self.client or not contexts:
            return {
                "query": query,
                "ocr_text": ocr_text,
                "found": len(contexts) > 0,
                "results": contexts,
                "generated_response": None
            }
        
        # 2. 컨텍스트 포맷팅
        context_text = self.format_context(contexts)
        
        # 3. 프롬프트 구성
        prompt = self._build_prompt(query, ocr_text, context_text)
        
        # 4. Claude API 호출
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            generated_text = message.content[0].text
            
        except Exception as e:
            print(f"Claude API 오류: {e}")
            generated_text = "응답 생성 중 오류가 발생했습니다."
        
        # 5. 결과 반환
        return {
            "query": query,
            "ocr_text": ocr_text,
            "found": len(contexts) > 0,
            "results": contexts,
            "generated_response": generated_text
        }
    
    def generate_usage_guide(self, medicine_data: Dict) -> Dict:
        """
        식별된 의약품 정보를 바탕으로 복용 방법과 주의사항에 특화된 가이드를 생성합니다.
        """
        medicine_name = medicine_data.get("name", "")
        
        if not medicine_name:
            return {
                "error": "의약품 정보가 불충분합니다.",
                "generated_response": None
            }
        
        # 데이터베이스에서 상세 정보 검색
        contexts = self.retrieve_context(medicine_name)
        
        if not contexts:
            return {
                "medicine": medicine_data,
                "found": False,
                "message": "데이터베이스에서 해당 의약품 정보를 찾을 수 없습니다.",
                "generated_response": None
            }
        
        # Claude API를 사용할 수 없는 경우
        if not self.client:
            return {
                "medicine": medicine_data,
                "found": True,
                "results": contexts,
                "generated_response": None
            }
        
        # 컨텍스트 포맷팅
        context_text = self.format_context(contexts)
        
        # 복용 정보 및 주의사항 특화 프롬프트
        prompt = f"""당신은 의약품 복용 가이드 전문가입니다. 다음 의약품에 대해 환자가 꼭 알아야 할 복용 방법과 주의사항을 명확하고 친절하게 설명해주세요.

<의약품_정보>
{context_text}
</의약품_정보>

다음 항목들을 중심으로 설명해주세요:

## 1. 복용 방법
- 언제, 얼마나 복용해야 하는지
- 식전/식후 복용 여부
- 물과 함께 복용하는 방법
- 최대 복용량 및 간격

## 2. 주의사항
- 절대 함께 복용하면 안 되는 것들 (음식, 다른 약)
- 피해야 할 행동이나 상황
- 부작용이 나타날 수 있는 증상
- 특히 주의해야 할 대상 (임산부, 어린이, 노인 등)

## 3. 보관 방법
- 적절한 보관 온도와 장소
- 유통기한 확인 방법

## 4. 응급 상황
- 과다 복용 시 대처 방법
- 즉시 병원에 가야 하는 증상

**중요**: 
- 의학 용어는 쉬운 말로 풀어서 설명해주세요
- 환자가 실제로 따라할 수 있도록 구체적으로 설명해주세요
- 불확실한 정보는 "전문가와 상담하세요"라고 명확히 안내해주세요
"""
        
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            generated_text = message.content[0].text
            
        except Exception as e:
            print(f"Claude API 오류: {e}")
            generated_text = "응답 생성 중 오류가 발생했습니다."
        
        return {
            "medicine": medicine_data,
            "found": True,
            "results": contexts,
            "usage_guide": generated_text
        }
    
    def answer_question(self, medicine_data: Dict, user_question: str) -> Dict:
        """
        의약품 정보를 바탕으로 사용자의 구체적인 질문에 답변합니다.
        """
        medicine_name = medicine_data.get("name", "")
        
        if not medicine_name:
            return {
                "error": "의약품 정보가 불충분합니다.",
                "answer": None
            }
        
        # 데이터베이스에서 정보 검색
        contexts = self.retrieve_context(medicine_name)
        
        if not self.client:
            return {
                "question": user_question,
                "answer": "AI 답변 생성 기능을 사용할 수 없습니다. 검색된 정보를 참고하세요.",
                "contexts": contexts
            }
        
        context_text = self.format_context(contexts)
        
        # 질문에 맞춤형 답변 프롬프트
        prompt = f"""당신은 의약품 상담 전문가입니다. 환자의 질문에 정확하고 친절하게 답변해주세요.

<의약품_정보>
{context_text}
</의약품_정보>

<환자_질문>
{user_question}
</환자_질문>

**답변 가이드라인**:
1. 환자의 질문에 직접적으로 답변하세요
2. 의학 용어는 쉽게 풀어서 설명하세요
3. 구체적이고 실용적인 조언을 제공하세요
4. 데이터베이스에 없는 정보는 추측하지 말고, "전문가와 상담이 필요합니다"라고 안내하세요
5. 응급 상황이나 심각한 부작용이 의심되면 즉시 병원 방문을 권장하세요

답변:"""
        
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            answer = message.content[0].text
            
        except Exception as e:
            print(f"Claude API 오류: {e}")
            answer = "답변 생성 중 오류가 발생했습니다."
        
        return {
            "question": user_question,
            "medicine": medicine_data,
            "answer": answer,
            "contexts": contexts
        }
        """
        Claude API에 전달할 프롬프트를 구성합니다.
        """
        prompt = f"""당신은 의약품 정보 전문가입니다. 사용자가 촬영한 알약 이미지에서 추출한 정보를 바탕으로 의약품에 대해 설명해주세요.

<사용자_쿼리>
{query}
</사용자_쿼리>

"""
        
        if ocr_text:
            prompt += f"""<OCR_추출_텍스트>
{ocr_text}
</OCR_추출_텍스트>

"""
        
        prompt += f"""<데이터베이스_검색_결과>
{context}
</데이터베이스_검색_결과>

위 정보를 바탕으로 다음 형식으로 답변해주세요:

1. **약품명**: 정확한 약품명
2. **주요 효능**: 어떤 증상에 사용되는지
3. **주의사항**: 복용 시 주의해야 할 점
4. **추가 정보**: 성분, 용법용량 등

검색된 정보가 없거나 불확실한 경우, 반드시 "정확한 정보를 제공할 수 없습니다. 약사나 의사와 상담하시기 바랍니다."라고 안내해주세요.

의약품 정보는 정확성이 매우 중요하므로, 추측하지 말고 검색된 정보만을 기반으로 답변해주세요."""

        return prompt
