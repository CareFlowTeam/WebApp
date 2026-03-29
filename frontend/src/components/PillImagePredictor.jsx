import React, { useState } from 'react';

const PillImagePredictor = () => {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append('file', image);
      const res = await fetch('/analyze/pill-image', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error('예측 실패: ' + res.status);
      const data = await res.json();
      setResult(data.result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="pill-image-predictor">
      <h2>의약품 이미지 탐지</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleImageChange} />
        <button type="submit" disabled={loading || !image}>예측하기</button>
      </form>
      {loading && <div>예측 중...</div>}
      {error && <div style={{color:'red'}}>에러: {error}</div>}
      {result && (
        <div className="pill-result">
          <h3>예측 결과</h3>
          <ul>
            {result.predictions && result.predictions.map((pred, idx) => (
              <li key={idx}>
                {pred.label_text} ({pred.class_name}) - 확률: {(pred.probability*100).toFixed(2)}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default PillImagePredictor;
