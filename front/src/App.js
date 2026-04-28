import './App.css';
import { useCallback, useState } from 'react';

function App() {

  const [isRecording, setIsRecording] = useState(false);
  const [isChatOpen, setChatOpen] = useState(false);
  const [isMenuOpen, setMenuOpen] = useState(false);
  const [theme, setTheme] = useState('light');

  const [error, setError] = useState(null);


  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  const startRecording = useCallback(() => {setIsRecording(true)});

  const stopRecording = useCallback(() => setIsRecording(false));

  const handlePointerDown = (e) => {
    e.preventDefault();
    if (!isRecording) startRecording();
  };

  const handlePointerUp = (e) => {
    e.preventDefault();
    if (isRecording) stopRecording();
  };

  const handlePointerLeave = () => {
    if (isRecording) stopRecording();
  };

  const sendData = () => {
    const formData = new FormData();
    if (text) {
      formData.append('text', text);
    }

    fetch('https://localhost:3000/process', {
      method: 'POST',
      body: formData,
    })
    .then(res => res.json())
    .then(data => {
      alert('Answer:' + data.message);
    })
  }

  return (
    <div className="App">
      
      <div className="page-container" data-theme={theme}>
        {/* Центрированный контейнер */}
        <div className="center-container">
          {/* Кнопка с крестом */}
          <button
            className={`cross-button ${isRecording ? 'recording' : ''}`}
            onPointerDown={handlePointerDown}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerLeave}
            onPointerCancel={handlePointerUp}
          />

          {/* Кнопка чата */}
          <button
            className="chat-button"
            onClick={() => setChatOpen(!isChatOpen)}
            aria-label="Открыть чат"
          >
            💬
          </button>
        </div>
        
        {/* Окно чата */}
        {isChatOpen && (
          <div className="chat-overlay">
            <div className="chat-box">
              {/* Кнопка закрытия */}
              <button
                className="close-chat"
                onClick={() => setChatOpen(false)}
                aria-label="Закрыть чат"
              >
                &times;
              </button>
              {/* Сообщения */}
              <div className='messages'>Чат</div>
              {/* Ввод и отправка */}
              <div className="chat-footer">
                <input
                  type="text"
                  placeholder="Введите сообщение..."
                  className="chat-input"
                  onChange={(e) => setText(e.target.value)}
                />
                <button className="send-button" onClick={sendData}>Отправить</button>
              </div>
            </div>
          </div>
        )}

        {/* Навбар снизу */}
        <div className="bottom-navbar">
          <button
            className="nav-button"
            onClick={() => setMenuOpen(!isMenuOpen)}
          >
            📋
          </button>
        </div>

        {/* Боковое меню */}
        {isMenuOpen && (
          <div className="side-menu">
            {/* Кнопка для переключения темы */}
            <label className='switch'>
              <input type="checkbox" onClick={toggleTheme}></input>
              <span class="slider"></span>
            </label>
            <button
              className="menu-close-btn"
              onClick={() => setMenuOpen(!isMenuOpen)}
            >
              &times;
            </button>
            <div className="menu-content">
              <p>Здесь меню</p>
            </div>
          </div>
          )}
      </div>
    </div>
  );
}

export default App;
