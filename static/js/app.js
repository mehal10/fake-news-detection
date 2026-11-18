/**
 * News Analysis System — Core UI interactions
 */

(function () {
  'use strict';

  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebarOverlay');
  const desktopToggle = document.getElementById('sidebarToggle');
  const mobileToggle = document.getElementById('mobileSidebarToggle');

  // Desktop collapse
  if (desktopToggle && sidebar) {
    desktopToggle.addEventListener('click', function () {
      sidebar.classList.toggle('collapsed');
      // Persist preference
      const collapsed = sidebar.classList.contains('collapsed');
      try {
        localStorage.setItem('sidebarCollapsed', collapsed ? '1' : '0');
      } catch (e) {}
    });

    // Restore preference
    try {
      if (localStorage.getItem('sidebarCollapsed') === '1') {
        sidebar.classList.add('collapsed');
      }
    } catch (e) {}
  }

  // Mobile drawer
  function openMobileSidebar() {
    if (sidebar) sidebar.classList.add('mobile-open');
    if (overlay) overlay.classList.add('visible');
    document.body.style.overflow = 'hidden';
  }

  function closeMobileSidebar() {
    if (sidebar) sidebar.classList.remove('mobile-open');
    if (overlay) overlay.classList.remove('visible');
    document.body.style.overflow = '';
  }

  if (mobileToggle) {
    mobileToggle.addEventListener('click', openMobileSidebar);
  }

  if (overlay) {
    overlay.addEventListener('click', closeMobileSidebar);
  }

  // Close mobile sidebar on nav click
  if (sidebar) {
    sidebar.querySelectorAll('.nav-item').forEach(function (item) {
      item.addEventListener('click', function () {
        if (window.innerWidth <= 1024) {
          closeMobileSidebar();
        }
      });
    });
  }

  // Handle resize
  window.addEventListener('resize', function () {
    if (window.innerWidth > 1024) {
      closeMobileSidebar();
    }
  });
})();
