const form = document.querySelector("#loan-form");
if (form) {
  form.addEventListener("submit", () => {
    const button = form.querySelector("button[type='submit']");
    if (button) {
      button.disabled = true;
      button.innerHTML = "Making prediction… <span>↻</span>";
    }
  });
}
