const state = {
  dialogues: [],
  originalDialogues: [],
  selectedIndex: -1,
};

const elements = {
  datasetStatus: document.querySelector("#datasetStatus"),
  jsonPath: document.querySelector("#jsonPath"),
  loadPathButton: document.querySelector("#loadPathButton"),
  fileInput: document.querySelector("#fileInput"),
  searchInput: document.querySelector("#searchInput"),
  conditionFilter: document.querySelector("#conditionFilter"),
  semanticFilter: document.querySelector("#semanticFilter"),
  dialogueList: document.querySelector("#dialogueList"),
  dialogueTitle: document.querySelector("#dialogueTitle"),
  dialogueMeta: document.querySelector("#dialogueMeta"),
  emptyState: document.querySelector("#emptyState"),
  chatMessages: document.querySelector("#chatMessages"),
  resetDialogueButton: document.querySelector("#resetDialogueButton"),
  copyDialogueButton: document.querySelector("#copyDialogueButton"),
  downloadButton: document.querySelector("#downloadButton"),
};

elements.loadPathButton.addEventListener("click", () => {
  loadFromPath(elements.jsonPath.value.trim());
});

elements.fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    const text = await file.text();
    loadDataset(JSON.parse(text), file.name);
  } catch (error) {
    showStatus(`Could not read JSON file: ${error.message}`, true);
  }
});

elements.searchInput.addEventListener("input", renderDialogueList);
elements.conditionFilter.addEventListener("change", renderDialogueList);
elements.semanticFilter.addEventListener("change", renderDialogueList);
elements.resetDialogueButton.addEventListener("click", resetSelectedDialogue);
elements.copyDialogueButton.addEventListener("click", copySelectedDialogue);
elements.downloadButton.addEventListener("click", downloadEditedDataset);

async function loadFromPath(path) {
  if (!path) {
    showStatus("Enter a JSON path or use Open JSON file.", true);
    return;
  }
  try {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    loadDataset(await response.json(), path);
  } catch (error) {
    showStatus(`Could not load path. If opened as a file, use Open JSON file. ${error.message}`, true);
  }
}

function loadDataset(data, sourceName) {
  if (!Array.isArray(data)) {
    showStatus("Expected a JSON array of dialogues.", true);
    return;
  }

  state.dialogues = structuredClone(data);
  state.originalDialogues = structuredClone(data);
  state.selectedIndex = data.length ? 0 : -1;

  populateFilters();
  renderDialogueList();
  renderSelectedDialogue();
  showStatus(`${data.length} dialogues loaded from ${sourceName}`);
  elements.downloadButton.disabled = !data.length;
}

function populateFilters() {
  setSelectOptions(elements.conditionFilter, "All conditions", uniqueValues("condition"));
  setSelectOptions(elements.semanticFilter, "All variants", uniqueValues("semantic_variant"));
}

function uniqueValues(field) {
  return [...new Set(state.dialogues.map((dialogue) => dialogue[field]).filter(Boolean))].sort();
}

function setSelectOptions(select, allLabel, values) {
  select.innerHTML = "";
  const all = document.createElement("option");
  all.value = "";
  all.textContent = allLabel;
  select.append(all);
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  });
}

function renderDialogueList() {
  const filtered = filteredDialogues();
  elements.dialogueList.innerHTML = "";

  filtered.forEach(({ dialogue, index }) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `dialogue-item${index === state.selectedIndex ? " active" : ""}`;
    button.addEventListener("click", () => {
      state.selectedIndex = index;
      renderDialogueList();
      renderSelectedDialogue();
    });

    const name = document.createElement("span");
    name.className = "dialogue-name";
    name.textContent = dialogue.dialogue_id ?? `dialogue_${index + 1}`;

    const tags = document.createElement("span");
    tags.className = "dialogue-tags";
    [
      dialogue.condition,
      dialogue.semantic_variant,
      dialogue.expected_outcome,
      `${dialogue.messages?.length ?? 0} messages`,
    ]
      .filter(Boolean)
      .forEach((value) => {
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = value;
        tags.append(tag);
      });

    button.append(name, tags);
    elements.dialogueList.append(button);
  });

  if (!filtered.length) {
    const empty = document.createElement("p");
    empty.className = "empty-list";
    empty.textContent = state.dialogues.length ? "No dialogues match the filters." : "No dataset loaded.";
    elements.dialogueList.append(empty);
  }
}

function filteredDialogues() {
  const query = elements.searchInput.value.trim().toLowerCase();
  const condition = elements.conditionFilter.value;
  const semanticVariant = elements.semanticFilter.value;

  return state.dialogues
    .map((dialogue, index) => ({ dialogue, index }))
    .filter(({ dialogue }) => !condition || dialogue.condition === condition)
    .filter(({ dialogue }) => !semanticVariant || dialogue.semantic_variant === semanticVariant)
    .filter(({ dialogue }) => {
      if (!query) return true;
      const text = [
        dialogue.dialogue_id,
        dialogue.case_id,
        dialogue.condition,
        dialogue.semantic_variant,
        dialogue.expected_outcome,
        ...(dialogue.messages ?? []).map((message) => message.content),
      ]
        .join("\n")
        .toLowerCase();
      return text.includes(query);
    });
}

function renderSelectedDialogue() {
  const dialogue = state.dialogues[state.selectedIndex];
  elements.chatMessages.innerHTML = "";
  const hasDialogue = Boolean(dialogue);
  elements.emptyState.style.display = hasDialogue ? "none" : "grid";
  elements.chatMessages.classList.toggle("visible", hasDialogue);
  elements.resetDialogueButton.disabled = !hasDialogue;
  elements.copyDialogueButton.disabled = !hasDialogue;

  if (!dialogue) {
    elements.dialogueTitle.textContent = "Load a dataset";
    elements.dialogueMeta.textContent = "Select a dialogue to preview and edit messages.";
    return;
  }

  elements.dialogueTitle.textContent = dialogue.dialogue_id ?? `Dialogue ${state.selectedIndex + 1}`;
  elements.dialogueMeta.textContent = [
    dialogue.case_id,
    dialogue.condition,
    dialogue.semantic_variant,
    dialogue.expected_outcome,
  ]
    .filter(Boolean)
    .join(" | ");

  (dialogue.messages ?? []).forEach((message, messageIndex) => {
    elements.chatMessages.append(createMessageEditor(message, messageIndex));
  });
}

function createMessageEditor(message, messageIndex) {
  const row = document.createElement("article");
  row.className = `message-row ${message.role === "user" ? "user" : "assistant"}`;

  const toolbar = document.createElement("div");
  toolbar.className = "message-toolbar";

  const roleSelect = document.createElement("select");
  ["user", "assistant"].forEach((role) => {
    const option = document.createElement("option");
    option.value = role;
    option.textContent = role;
    option.selected = message.role === role;
    roleSelect.append(option);
  });
  roleSelect.addEventListener("change", () => {
    message.role = roleSelect.value;
    renderSelectedDialogue();
    renderDialogueList();
  });

  const index = document.createElement("span");
  index.className = "message-index";
  index.textContent = `#${messageIndex + 1}`;

  const actions = document.createElement("div");
  actions.className = "message-actions";

  const addButton = document.createElement("button");
  addButton.type = "button";
  addButton.textContent = "+";
  addButton.title = "Add message after this one";
  addButton.addEventListener("click", () => addMessageAfter(messageIndex));

  const deleteButton = document.createElement("button");
  deleteButton.type = "button";
  deleteButton.textContent = "Delete";
  deleteButton.className = "danger";
  deleteButton.addEventListener("click", () => deleteMessage(messageIndex));

  actions.append(addButton, deleteButton);
  toolbar.append(roleSelect, index, actions);

  const textarea = document.createElement("textarea");
  textarea.className = "message-editor";
  textarea.value = message.content ?? "";
  textarea.addEventListener("input", () => {
    message.content = textarea.value;
    autoSizeTextarea(textarea);
  });
  requestAnimationFrame(() => autoSizeTextarea(textarea));

  row.append(toolbar, textarea);
  return row;
}

function autoSizeTextarea(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${textarea.scrollHeight + 2}px`;
}

function addMessageAfter(messageIndex) {
  const dialogue = state.dialogues[state.selectedIndex];
  if (!dialogue) return;
  const currentRole = dialogue.messages[messageIndex]?.role;
  const nextRole = currentRole === "user" ? "assistant" : "user";
  dialogue.messages.splice(messageIndex + 1, 0, {
    role: nextRole,
    content: "",
  });
  renderSelectedDialogue();
  renderDialogueList();
}

function deleteMessage(messageIndex) {
  const dialogue = state.dialogues[state.selectedIndex];
  if (!dialogue) return;
  dialogue.messages.splice(messageIndex, 1);
  renderSelectedDialogue();
  renderDialogueList();
}

function resetSelectedDialogue() {
  if (state.selectedIndex < 0) return;
  state.dialogues[state.selectedIndex] = structuredClone(state.originalDialogues[state.selectedIndex]);
  renderSelectedDialogue();
  renderDialogueList();
}

async function copySelectedDialogue() {
  const dialogue = state.dialogues[state.selectedIndex];
  if (!dialogue) return;
  await navigator.clipboard.writeText(JSON.stringify(dialogue, null, 2));
  showStatus("Selected dialogue copied.");
}

function downloadEditedDataset() {
  const blob = new Blob([JSON.stringify(state.dialogues, null, 2) + "\n"], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "dialogues_edited.json";
  link.click();
  URL.revokeObjectURL(url);
  showStatus("Edited dataset exported.");
}

function showStatus(message, isError = false) {
  elements.datasetStatus.textContent = message;
  elements.datasetStatus.style.color = isError ? "#b42318" : "";
}
