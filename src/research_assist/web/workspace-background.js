/**
 * Antigravity Interactive Background for ResearchDesk (web_demo2)
 * Ported directly from antigravity-basic.html
 * - Upward floating geometric particles (Circle, Square, Triangle)
 * - Mouse repulsion / physics interaction
 * - Zero impact on UI events (pointer-events: none)
 */

(function () {
  const canvas = document.getElementById("antigravity-bg");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let animationFrame;
  let width, height;

  const config = {
    particleCount: 65,
    speedFactor: 0.8,
    // Google Palette from antigravity-basic.html
    colors: ["#EA4335", "#FBBC05", "#34A853", "#4285F4", "#E8F0FE", "#DADCE0"],
    shape: "mixed", // circle, square, triangle, mixed
    gravity: -0.05, // Upward antigravity flow
    interactionRadius: 150,
    friction: 0.98,
  };

  // High DPI scaling
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    width = window.innerWidth;
    height = window.innerHeight;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);
  }

  window.addEventListener("resize", resize);
  resize();

  class Particle {
    constructor() {
      this.init(true);
    }

    init(randomY = false) {
      this.x = Math.random() * width;
      // If randomY is true, scatter across screen (initial load)
      // Otherwise spawn below bottom (for antigravity upward flow)
      this.y = randomY ? Math.random() * height : height + 50;

      this.size = Math.random() * 14 + 6;
      this.targetSize = this.size;

      // Random initial velocity
      this.vx = (Math.random() - 0.5) * 2 * config.speedFactor;
      this.vy = (Math.random() - 0.5) * 2 * config.speedFactor;

      // Inherent upward bias
      if (config.gravity < 0) {
        this.vy -= Math.random() * 1 * Math.abs(config.gravity);
      } else {
        this.y = randomY ? Math.random() * height : -50;
      }

      this.color =
        config.colors[Math.floor(Math.random() * config.colors.length)];
      this.rotation = Math.random() * Math.PI * 2;
      this.rotationSpeed = (Math.random() - 0.5) * 0.05;

      // 0 = Circle, 1 = Square, 2 = Triangle
      this.type = this.getShapeType();

      // 3D depth factor (0.5 to 1.5)
      this.depth = Math.random() * 1 + 0.5;
    }

    getShapeType() {
      if (config.shape === "circle") return 0;
      if (config.shape === "square") return 1;
      if (config.shape === "triangle") return 2;
      return Math.floor(Math.random() * 3);
    }

    update(mouseX, mouseY) {
      // Apply gravity/antigravity
      this.vy += config.gravity * 0.05 * this.depth;

      // Movement
      this.x += this.vx * this.depth;
      this.y += this.vy * this.depth;
      this.rotation += this.rotationSpeed;

      // Mouse Interaction (Repel)
      const dx = this.x - mouseX;
      const dy = this.y - mouseY;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < config.interactionRadius) {
        const force =
          (config.interactionRadius - dist) / config.interactionRadius;
        const angle = Math.atan2(dy, dx);
        const push = force * 4;

        this.vx += Math.cos(angle) * push;
        this.vy += Math.sin(angle) * push;
      }

      // Friction
      this.vx *= config.friction;
      this.vy *= config.friction;

      // Side bounds wrap-around
      if (this.x < -50) this.x = width + 50;
      if (this.x > width + 50) this.x = -50;

      // Vertical reset
      if (config.gravity < 0) {
        if (this.y < -60) this.init(false);
      } else {
        if (this.y > height + 60) this.init(false);
      }
    }

    draw() {
      ctx.save();
      ctx.translate(this.x, this.y);
      ctx.rotate(this.rotation);
      ctx.fillStyle = this.color;

      // Soft opacity for pleasant background contrast
      ctx.globalAlpha =
        this.color === "#E8F0FE" || this.color === "#DADCE0" ? 0.9 : 0.7;

      // Subtle shadow for depth
      ctx.shadowColor = "rgba(0, 0, 0, 0.05)";
      ctx.shadowBlur = 8 * this.depth;
      ctx.shadowOffsetX = 3;
      ctx.shadowOffsetY = 3;

      const s = this.size * this.depth;

      ctx.beginPath();
      if (this.type === 0) {
        ctx.arc(0, 0, s / 2, 0, Math.PI * 2);
      } else if (this.type === 1) {
        ctx.rect(-s / 2, -s / 2, s, s);
      } else if (this.type === 2) {
        ctx.moveTo(0, -s / 2);
        ctx.lineTo(s / 2, s / 2);
        ctx.lineTo(-s / 2, s / 2);
        ctx.closePath();
      } else {
        ctx.rect(-s / 2, -s / 2, s, s);
      }

      ctx.fill();
      ctx.restore();
    }
  }

  const particles = [];
  for (let i = 0; i < config.particleCount; i++) {
    particles.push(new Particle());
  }

  let mouseX = -1000;
  let mouseY = -1000;

  window.addEventListener("mousemove", (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
  });

  window.addEventListener("mouseleave", () => {
    mouseX = -1000;
    mouseY = -1000;
  });

  // Touch support for mobile/touch screens
  window.addEventListener(
    "touchmove",
    (e) => {
      if (e.touches && e.touches.length > 0) {
        mouseX = e.touches[0].clientX;
        mouseY = e.touches[0].clientY;
      }
    },
    { passive: true },
  );

  window.addEventListener("touchend", () => {
    mouseX = -1000;
    mouseY = -1000;
  });

  function animate() {
    ctx.clearRect(0, 0, width, height);

    for (let i = 0; i < particles.length; i++) {
      particles[i].update(mouseX, mouseY);
      particles[i].draw();
    }

    animationFrame = requestAnimationFrame(animate);
  }

  function updateAnimation() {
    cancelAnimationFrame(animationFrame);
    if (!document.hidden && !reducedMotion.matches) animate();
    else ctx.clearRect(0, 0, width, height);
  }
  document.addEventListener("visibilitychange", updateAnimation);
  reducedMotion.addEventListener("change", updateAnimation);
  updateAnimation();
})();
