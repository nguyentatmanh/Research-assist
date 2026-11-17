export function downloadText(filename, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  try {
    link.click();
  } finally {
    link.remove();
    // Allow the browser to finish opening the download before releasing its URL.
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
