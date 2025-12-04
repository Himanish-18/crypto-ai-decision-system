import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'

const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
    },
})

export const auth = {
    login: async (username: string, password: string) => {
        const params = new URLSearchParams()
        params.append('username', username)
        params.append('password', password)
        const { data } = await api.post('/login/access-token', params)
        return data
    },
    register: async (email: string, password: string) => {
        const { data } = await api.post('/register', { email, password })
        return data
    }
}

export default api
