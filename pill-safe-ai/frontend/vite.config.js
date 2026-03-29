import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  // tailwindcss 플러그인을 유지해야 디자인이 깨지지 않습니다.
  plugins: [react(), tailwindcss()],
  server: {
    host: true,
    strictPort: true,
    proxy: {
      // 1. FastAPI 통합 백엔드
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
        // /api/search -> /search, /api/pharmacies/status -> /pharmacies/status
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      // 2. FastAPI 별도 네임스페이스 유지
      '/ml': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/ml/, ''),
      },
    },
  },
})