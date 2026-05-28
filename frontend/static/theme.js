/* ================================================================
   NEXUS CV — Theme Engine (No-Flicker Dark Mode)
   Must be loaded in <head> BEFORE body renders
   ================================================================ */

// Apply theme IMMEDIATELY to prevent flash
(function() {
    const saved = localStorage.getItem('nexuscv-theme');
    if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        document.documentElement.classList.add('dark');
    }
})();

function applyTheme() {
    const saved = localStorage.getItem('nexuscv-theme');
    if (saved === 'dark') {
        document.body.classList.add('dark');
    } else if (saved === 'light') {
        document.body.classList.remove('dark');
    } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
        document.body.classList.add('dark');
    }
}

function toggleTheme() {
    const isDark = document.body.classList.toggle('dark');
    document.documentElement.classList.toggle('dark', isDark);
    localStorage.setItem('nexuscv-theme', isDark ? 'dark' : 'light');
}

function toggleMenu() {
    const menu = document.getElementById('dropdownMenu');
    if (menu) {
        menu.classList.toggle('show');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(e) {
    const menu = document.getElementById('dropdownMenu');
    const avatar = document.querySelector('.avatar');
    if (menu && menu.classList.contains('show') && !menu.contains(e.target) && e.target !== avatar) {
        menu.classList.remove('show');
    }
});

// Mobile nav toggle
function toggleMobileNav() {
    const links = document.querySelector('.nav-links');
    if (links) {
        links.classList.toggle('mobile-open');
    }
}

// Close mobile nav on link click
document.addEventListener('DOMContentLoaded', function() {
    applyTheme();
    document.querySelectorAll('.nav-links a').forEach(function(a) {
        a.addEventListener('click', function() {
            const links = document.querySelector('.nav-links');
            if (links) links.classList.remove('mobile-open');
        });
    });
});
