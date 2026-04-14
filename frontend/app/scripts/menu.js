const menuToggle = document.getElementById('menuToggle');
const menuPanel = document.getElementById('menuPanel');
const settingsMobile = document.getElementById('settingsMobile');
const settingsListMobile = document.getElementById('settingsListMobile');

// Открытие/закрытие меню по бургеру
menuToggle.addEventListener('click', () => {
  const isVisible = menuPanel.style.display === 'flex';
  menuPanel.style.display = isVisible ? 'none' : 'flex';
  // Обновляем aria
  menuToggle.setAttribute('aria-expanded', String(!isVisible));
});

// Клик вне меню закрывает его
document.addEventListener('click', (e) => {
  if (!e.target.closest('.header') && !e.target.closest('.menu-panel')) {
    menuPanel.style.display = 'none';
    // Обновляем aria
    menuToggle.setAttribute('aria-expanded', 'false');
    // Также скрываем подсписок
    if (settingsListMobile) {
      settingsListMobile.style.display = 'none';
    }
  }
});

// Клик по "Настройки" – показываем/скрываем подсписок
settingsMobile.addEventListener('click', (e) => {
  e.stopPropagation();
  if (!settingsListMobile || settingsListMobile.style.display === 'block') {
    settingsListMobile.style.display = 'none';
  } else {
    settingsListMobile.style.display = 'block';
  }
});