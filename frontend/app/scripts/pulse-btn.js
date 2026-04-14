const button = document.getElementById('heartBtn');

button.addEventListener('mousedown', () => {
  button.classList.add('pulse');
});
button.addEventListener('mouseup', () => {
  button.classList.remove('pulse');
});
button.addEventListener('mouseleave', () => {
  button.classList.remove('pulse');
});