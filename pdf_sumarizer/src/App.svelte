<script>
  let file = null;
  let fileInput;
  let summary = "";
  let question = "";
  let answer = "";
  let loadingSummary = false;
  let loadingAnswer = false;

  function handleFileChange() {
      file = fileInput.files[0];
  }

  async function uploadFile() {
      if (!file) return alert("Please select a file");
      loadingSummary = true;
      summary = "";

      let formData = new FormData();
      formData.append("file", file);

      try {
          let res = await fetch("http://127.0.0.1:5000/summarize", {
              method: "POST",
              body: formData
          });

          let result = await res.json();
          summary = result.summary || result.error;
      } catch (error) {
          summary = "Error summarizing the file.";
      }

      loadingSummary = false;
  }

  async function askQuestion() {
      if (!question) return alert("Please enter a question");
      loadingAnswer = true;
      answer = "";

      try {
          let res = await fetch("http://127.0.0.1:5000/ask", {
              method: "POST",
              body: JSON.stringify({ question }),
              headers: { "Content-Type": "application/json" }
          });

          let result = await res.json();
          answer = result.answer || result.error;
      } catch (error) {
          answer = "Error processing the question.";
      }

      loadingAnswer = false;
  }
</script>

<main class="container">
  <h1>📄 PDF Summarizer & Q&A</h1>

  <div class="upload-section">
      <label> Upload PDF:</label>
      <input type="file" bind:this={fileInput} on:change={handleFileChange} accept=".pdf" />
      <button on:click={uploadFile} disabled={loadingSummary}>
          {loadingSummary ? "Summarizing..." : "Summarize"}
      </button>
  </div>

  {#if summary}
      <div class="summary-section">
          <h2>📌 Summary:</h2>
          <p>{summary}</p>
      </div>
  {/if}

  <hr />

  {#if summary}
      <div class="qa-section">
          <label> Ask a Question:</label>
          <input type="text" bind:value={question} placeholder="Type your question..." />
          <button on:click={askQuestion} disabled={loadingAnswer}>
              {loadingAnswer ? "Thinking..." : "Ask"}
          </button>

          {#if answer}
              <h2>🗨️ Answer:</h2>
              <p>{answer}</p>
          {/if}
      </div>
  {/if}
</main>

<style>
  .container {
      max-width: 600px;
      margin: auto;
      text-align: center;
      font-family: Arial, sans-serif;
      background: #f9f9f9;
      padding: 20px;
      border-radius: 10px;
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  }

  .upload-section, .qa-section {
      margin: 20px 0;
  }

  input, button {
      margin: 10px;
      padding: 10px;
      width: 80%;
      border-radius: 5px;
      border: 1px solid #ccc;
  }

  button {
      background-color: #007bff;
      color: white;
      cursor: pointer;
  }

  button:disabled {
      background-color: #cccccc;
      cursor: not-allowed;
  }

  h1 {
      color: #333;
  }

  .summary-section, .qa-section {
      background: white;
      padding: 15px;
      border-radius: 5px;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }
</style>
