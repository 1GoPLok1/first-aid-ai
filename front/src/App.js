import logo from './logo.svg';
import './App.css';

function App() {

  return (
    <div className="App">
      {/* Навигация */}
      <div className="navbar">
        <button className="menu-button">
          меню ▼
        </button>
      </div>

      {/* Основной контейнер */}
      <div className="container">
        <div className="chat-container">
          
          {/* Чат с вводом и кнопкой */}
          <div className="chat-box">
            <input
              className="chat-input"
              type="text"
              placeholder="Введите сообщение..."
            />
            <button className="send-button">
              Отправить
            </button>
          </div>

        </div>
      </div>

      {/* Подвал */}
      <div className="footer">
        <div className="social-info">
          <p>
            Следите за нами:{" "}
            <a href="https://twitter.com" target="_blank" rel="noopener noreferrer">Twitter</a>,
            <a href="https://facebook.com" target="_blank" rel="noopener noreferrer">Facebook</a>,
            <a href="https://instagram.com" target="_blank" rel="noopener noreferrer">Instagram</a>
          </p>
          <p>Автор: Иван Иванов</p>
        </div>
      </div>
    </div>
  );
}

export default App;
