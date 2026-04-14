const chatToggle = document.getElementById('chatToggle');
const chatForm = document.getElementById('chatForm');
const closeBtn = document.getElementById('closeChat');

chatToggle.addEventListener('click', () => {
  chatForm.style.display = 'block'; // показываем форму
});

closeBtn.addEventListener('click', () => {
  chatForm.style.display = 'none'; // скрываем форму
});