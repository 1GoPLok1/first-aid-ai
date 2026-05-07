
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export class AssistantAPI {
  static async sendQuery(query, category = null, sessionId = null) {
    const payload = {
      query: query.trim(),
      category: category,
      session_id: sessionId || this.generateSessionId(),
      top_k: 5
    };

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return {
        success: true,
         data,
        sessionId: payload.session_id
      };

    } catch (error) {
      console.error('API request failed:', error);
      return {
        success: false,
        error: error.message,
        sessionId: payload.session_id
      };
    }
  }

  static generateSessionId() {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}