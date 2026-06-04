import React, { useState, useEffect, useRef } from 'react';
import {
  ConfigProvider, Layout, List, Button, Select, Input, Card,
  Typography, Collapse, message, Space, Tag, Switch, Slider, Drawer,
  theme as antdTheme
} from 'antd';
import {
  MenuOutlined, SettingOutlined, PlusOutlined, 
  SendOutlined, AudioOutlined, StopOutlined, FileTextOutlined
} from '@ant-design/icons';
import './App.css';
import { ignore } from 'antd/es/theme/useToken';

const { Text, Link } = Typography;
const { TextArea } = Input;

const App = () => {
  const [themeType, setThemeType] = useState('first_aid');
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [healthStatus, setHealthStatus] = useState({ groq: false, qdrant: false });
  
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [uiSettings, setUiSettings] = useState({ darkMode: true, fontSize: 16 });

  const chatEndRef = useRef(null);
  const abortControllerRef = useRef(null);
  const recognitionRef = useRef(null);

  const [activeFont, setActiveFont] = useState('Montserrat');

  useEffect(() => {
    fetch('/api/v1/sessions').then(r => r.json()).then(setSessions).catch(() => message.error('Не удалось загрузить историю'));
  }, []);

  useEffect(() => {
    fetch('/api/v1/health').then(r => r.json()).then(setHealthStatus).catch(() => setHealthStatus({ groq: false, qdrant: false }));
  }, []);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const antdConfig = {
    algorithm: uiSettings.darkMode ? antdTheme.darkAlgorithm : antdTheme.defaultAlgorithm,
    token: {
      fontFamily: `'${activeFont}', system-ui, sans-serif`,
      fontSize: uiSettings.fontSize,
      colorBgBase: uiSettings.darkMode ? '#100E11' : '#F8F7F5',
      colorBgContainer: uiSettings.darkMode ? '#1A171A' : '#FFFFFF',
      colorBgElevated: uiSettings.darkMode ? '#2A2527' : '#F0EFED',
      colorBorder: '#706B68',
      colorText: uiSettings.darkMode ? '#E8E0DC' : '#1A1A1A',
      colorTextSecondary: uiSettings.darkMode ? '#C0B5B3' : '#706B68',
      colorPrimary: '#490206',
      colorPrimaryHover: '#2C0001',
      colorError: '#EF4444',
      colorSuccess: '#22C55E',
      borderRadius: 30,
    }
  };

  const handleNewSession = async () => {
    try {
      const res = await fetch('/api/v1/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ theme: themeType })
      });
      const data = await res.json();
      setCurrentSessionId(data.id);
      setMessages([]);
      message.success('Создан новый диалог');
    } catch { message.error('Ошибка создания сессии'); }
  };

  const handleSend = async () => {
    if (!inputValue.trim() || isStreaming || !currentSessionId) {
      if (!currentSessionId) message.warning('Сначала создайте новый диалог');
      return;
    }
    const userMsg = { role: 'user', content: inputValue };
    setMessages(prev => [...prev, userMsg]);
    setInputValue('');
    setIsStreaming(true);
    setMessages(prev => [...prev, { role: 'assistant', content: '', sources: [], isTyping: true }]);

    abortControllerRef.current = new AbortController();
    try {
      const response = await fetch('/api/v1/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: currentSessionId, message: userMsg.content, theme: themeType }),
        signal: abortControllerRef.current.signal
      });
      if (!response.ok) throw new Error('Network error');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '', accText = '', accSources = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.token) accText += data.token;
            if (data.sources) accSources = data.sources;
            setMessages(prev => {
              const upd = [...prev];
              const i = upd.length - 1;
              if (upd[i]?.role === 'assistant') upd[i] = { ...upd[i], content: accText, sources: accSources, isTyping: false };
              return upd;
            });
          } catch {}
        }
      }
    } catch (err) { if (err.name !== 'AbortError') message.error('Ошибка получения ответа'); }
    finally { setIsStreaming(false); abortControllerRef.current = null; }
  };

  const toggleVoiceInput = () => {
    if (isRecording) { recognitionRef.current?.stop(); setIsRecording(false); return; }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) { message.error('Браузер не поддерживает распознавание речи'); return; }
    const rec = new SpeechRecognition();
    rec.lang = 'ru-RU'; rec.interimResults = false; rec.continuous = false;
    rec.onstart = () => setIsRecording(true);
    rec.onresult = (e) => setInputValue(p => p + (p ? ' ' : '') + e.results[0][0].transcript);
    rec.onend = () => setIsRecording(false);
    rec.onerror = () => { setIsRecording(false); message.error('Ошибка распознавания'); };
    recognitionRef.current = rec; rec.start();
  };

  return (
    <ConfigProvider theme={antdConfig}>
      <div className="app-root" data-theme={uiSettings.darkMode ? 'dark' : 'light'} style={{ '--fs': `${uiSettings.fontSize}px` }}>
        
        {/* 🔝 TOPBAR */}
        <header className="top-bar">
          <div className="tb-left">
            <Button type="text" icon={<MenuOutlined />} onClick={() => setSidebarOpen(true)} className="icon-btn" />
          </div>
          <div className="tb-center">
            <h1 className="app-title">AI Первая помощь</h1>
          </div>
          <div className="tb-right">
            <div className="status-group">
              <div className={`status-item ${healthStatus.groq ? 'on' : 'off'}`}>
                <span className="status-dot" />
                <span className="status-label">Groq</span>
              </div>
              <div className={`status-item ${healthStatus.qdrant ? 'on' : 'off'}`}>
                <span className="status-dot" />
                <span className="status-label">Qdrant</span>
              </div>
            </div>
            <Button type="text" icon={<SettingOutlined />} onClick={() => setSettingsOpen(true)} className="icon-btn" />
          </div>
        </header>

        {/* 📄 MAIN CHAT */}
        <main className="chat-main">
          <div className="chat-scroll" ref={chatEndRef}>
            {messages.map((msg, i) => (
              <div key={i} className={`msg-row ${msg.role}`}>
                <Card size="small" className={`msg-card ${msg.role}`}>
                  <div className="msg-text">{msg.content}</div>
                  {msg.sources?.length > 0 && (
                    <Collapse ghost style={{ marginTop: 6 }} expandIconPosition="end">
                      <Collapse.Panel header={<Space><FileTextOutlined /> Источники</Space>} key="src">
                        {msg.sources.map((s, idx) => (
                          <div key={idx} style={{ marginBottom: 4 }}>
                            <Link href={s.url} target="_blank">{s.title}</Link>
                            <Text type="secondary" style={{ marginLeft: 6 }}>— стр. {s.page}</Text>
                          </div>
                        ))}
                      </Collapse.Panel>
                    </Collapse>
                  )}
                  {msg.isTyping && <div className="typing"><span/><span/><span/></div>}
                </Card>
              </div>
            ))}
          </div>

          <div className="input-bar">
            <TextArea
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              onPressEnter={e => { if (!e.shiftKey) { e.preventDefault(); handleSend(); } }}
              placeholder="Опишите симптомы..."
              autoSize={{ minRows: 1, maxRows: 4 }}
              className="chat-input"
            />
            <Button icon={isRecording ? <StopOutlined /> : <AudioOutlined />} onClick={toggleVoiceInput} danger={isRecording} className="icon-btn voice" />
            <Button type="primary" icon={isStreaming ? <StopOutlined /> : <SendOutlined />} onClick={isStreaming ? () => abortControllerRef.current?.abort() : handleSend} disabled={isRecording} className="icon-btn send" />
          </div>
        </main>

        {/* 🚨 DISCLAIMER */}
        <footer className="disclaimer">⚠️ Система носит рекомендательный характер. При угрозе жизни звоните 112</footer>

        {/* 📑 DRAWER: НАВИГАЦИЯ */}
        <Drawer title="Навигация" placement="left" onClose={() => setSidebarOpen(false)} open={sidebarOpen} width={280} styles={{ body: { padding: '16px', display: 'flex', flexDirection: 'column', height: 'calc(100vh - 50px)' } }}>
          <div className="sidebar-top">
            <Select value={themeType} onChange={setThemeType} style={{ width: '100%', marginBottom: 16 }} options={[{ value: 'first_aid', label: '💊 Первая помощь' }, { value: 'healthy_lifestyle', label: '🥗 ЗОЖ и профилактика' }]} />
          </div>
          <div className="sidebar-bottom">
            <Button type="primary" icon={<PlusOutlined />} onClick={handleNewSession} block style={{ marginBottom: 12 }}>Новый диалог</Button>
            <Text strong style={{ marginBottom: 8, display: 'block' }}>История диалогов</Text>
            <div className="history-scroll">
              <List size="small" bordered dataSource={sessions} renderItem={s => (
                <List.Item onClick={() => { setCurrentSessionId(s.id); setMessages([]); setSidebarOpen(false); }} className={`history-item ${s.id === currentSessionId ? 'active' : ''}`}>
                  <Text ellipsis>{s.title || `Диалог от ${new Date(s.created_at).toLocaleDateString()}`}</Text>
                </List.Item>
              )} />
            </div>
          </div>
        </Drawer>

        {/* ⚙️ DRAWER: НАСТРОЙКИ */}
        <Drawer title="Настройки" placement="right" onClose={() => setSettingsOpen(false)} open={settingsOpen} width={300}>
          <Space direction="vertical" style={{ width: '100%' }} size="large">
            <div className="setting-row"><span>Тёмная тема</span><Switch checked={uiSettings.darkMode} onChange={v => setUiSettings(p => ({ ...p, darkMode: v }))} /></div>
            <div className="setting-row">
              <span>Шрифт интерфейса</span>
              <Select 
                style={{ width: 180 }} 
                value={activeFont} 
                onChange={setActiveFont}
                options={[
                  { value: 'Montserrat', label: 'Montserrat' },
                  { value: 'Montserrat Alternates', label: 'Alternates' }
                ]} 
              />
            </div>
            <div className="setting-row-column">
              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Размер шрифта</span><Text type="secondary">{uiSettings.fontSize}px</Text></div>
              <Slider min={14} max={22} step={1} value={uiSettings.fontSize} onChange={v => setUiSettings(p => ({ ...p, fontSize: v }))} />
            </div>
            <Button onClick={() => setSettingsOpen(false)} block type="primary">Готово</Button>
          </Space>
        </Drawer>
      </div>
    </ConfigProvider>
  );
};

export default App;