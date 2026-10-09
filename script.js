const textInput = document.getElementById("textInput");
const charCount = document.getElementById("charCount");
const analyzeBtn = document.getElementById("analyzeBtn");
const errorBox = document.getElementById("errorBox");
const resultSection = document.getElementById("resultSection");

let sentimentChart = null;
let emotionChart = null;

textInput.addEventListener("input", () => {
  charCount.textContent = `${textInput.value.length} / 5000`;
});

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}
function hideError() {
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
}

function renderSentimentChart(probabilities) {
  const ctx = document.getElementById("sentimentChart");
  if (sentimentChart) sentimentChart.destroy();

  sentimentChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Positive", "Neutral", "Negative"],
      datasets: [{
        label: "Confidence (%)",
        data: [probabilities.positive, probabilities.neutral, probabilities.negative],
        borderRadius: 9,
        backgroundColor: ["#62c88b", "#f2b84b", "#ed7474"],
        borderWidth: 0
      }]
    },
    options: {
      responsive:true, maintainAspectRatio:false,
      scales:{y:{beginAtZero:true,max:100,ticks:{callback:value=>value+"%"}},x:{grid:{display:false}}},
      plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>` ${c.raw}%`}}}
    }
  });
}

function renderEmotionChart(probabilities) {
  const ctx = document.getElementById("emotionChart");
  if (emotionChart) emotionChart.destroy();

  const labels = ["Happy","Sad","Angry","Frustrated","Fearful","Surprised","Love","Disappointed","Excited","Neutral"];
  const keys = ["happy","sad","angry","frustrated","fearful","surprised","love","disappointed","excited","neutral"];

  emotionChart = new Chart(ctx, {
    type:"bar",
    data:{
      labels,
      datasets:[{
        label:"Confidence (%)",
        data:keys.map(key=>probabilities[key] ?? 0),
        borderRadius:7,
        backgroundColor:["#62c88b","#8298e8","#ed7474","#e58b55","#9b7bd3","#e7ad4d","#ef6fa5","#8e91a0","#5ca9d6","#b4bbc8"],
        borderWidth:0
      }]
    },
    options:{
      indexAxis:"y", responsive:true, maintainAspectRatio:false,
      scales:{x:{beginAtZero:true,max:100,ticks:{callback:value=>value+"%"}},y:{grid:{display:false}}},
      plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>` ${c.raw}%`}}}
    }
  });
}

async function analyze() {
  const text = textInput.value.trim();
  hideError();

  if (!text) {
    showError("Please enter some text first.");
    textInput.focus();
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = "Analyzing…";

  try {
    const response = await fetch("/api/analyze", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({text})
    });
    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Unable to analyze the text.");
    }

    document.getElementById("userText").textContent = `“${data.text}”`;

    document.getElementById("sentimentEmoji").textContent = data.sentiment_emoji;
    document.getElementById("sentimentLabel").textContent = data.display_sentiment.toUpperCase();
    document.getElementById("sentimentConfidence").textContent = `${data.sentiment_confidence}%`;

    document.getElementById("emotionEmoji").textContent = data.emotion_emoji;
    document.getElementById("emotionLabel").textContent = data.display_emotion.toUpperCase();
    document.getElementById("emotionConfidence").textContent = `${data.emotion_confidence}%`;

    renderSentimentChart(data.sentiment_probabilities);
    renderEmotionChart(data.emotion_probabilities);

    resultSection.classList.remove("hidden");
    resultSection.scrollIntoView({behavior:"smooth",block:"start"});
  } catch(error) {
    showError(error.message || "Sorry, something went wrong. Please try again.");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = 'Analyze Sentiment <span>→</span>';
  }
}

analyzeBtn.addEventListener("click", analyze);
textInput.addEventListener("keydown", event => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") analyze();
});
