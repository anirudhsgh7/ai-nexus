import { useState, type FormEvent } from "react";

interface TaskFormProps {
  disabled: boolean;
  busy: boolean;
  error: string | null;
  onSubmit: (task: string) => void;
}

export function TaskForm({ disabled, busy, error, onSubmit }: TaskFormProps) {
  const [task, setTask] = useState("");
  const trimmed = task.trim();
  const canSubmit = !disabled && !busy && trimmed.length > 0;

  const handleSubmit = (event: FormEvent): void => {
    event.preventDefault();
    if (canSubmit) onSubmit(trimmed);
  };

  return (
    <form className="task-form" onSubmit={handleSubmit}>
      <label htmlFor="task-input">Task</label>
      <textarea
        id="task-input"
        value={task}
        rows={3}
        maxLength={4000}
        placeholder="What should the team work on?"
        disabled={disabled}
        onChange={(event) => setTask(event.target.value)}
      />
      <div className="form-row">
        <span className="char-count">{task.length}/4000</span>
        {error !== null ? (
          <span className="form-error" role="alert">
            {error}
          </span>
        ) : (
          <span />
        )}
        <button type="submit" disabled={!canSubmit}>
          {busy ? "Starting…" : "Run"}
        </button>
      </div>
    </form>
  );
}
