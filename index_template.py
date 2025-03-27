"""
This module contains the HTML template for the OpenAI Code interface.
The index_html variable can be imported and used in other files.

Example:
    from index_template import index_html
"""

index_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OpenAI Code</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                padding-top: 2rem;
                background-color: #f0f2f5;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
            .header {
                background: linear-gradient(90deg, #10a37f, #0d8a6f);
                color: white;
                padding: 1.5rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            .card {
                margin-bottom: 1.5rem;
                box-shadow: 0 6px 16px rgba(0,0,0,0.08);
                border: none;
                border-radius: 10px;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.12);
            }
            .card-header {
                background-color: #10a37f;
                color: white;
                font-weight: bold;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                padding: 1rem 1.25rem;
            }
            .model-badge {
                font-size: 0.85rem;
                padding: 0.35rem 0.75rem;
                border-radius: 20px;
                font-weight: 500;
            }
            .config-icon {
                font-size: 1.5rem;
                margin-right: 0.75rem;
            }
            .table-responsive {
                max-height: 500px;
                overflow-y: auto;
                border-radius: 8px;
            }
            .reasoning-note {
                font-size: 0.9rem;
                padding: 0.75rem;
                background-color: #e9f7f2;
                border-radius: 8px;
                border-left: 4px solid #10a37f;
                margin-bottom: 1.25rem;
            }
            .status-success {
                color: #10a37f;
                font-weight: 500;
            }
            .status-error {
                color: #dc3545;
                font-weight: 500;
            }
            .refresh-btn {
                font-size: 0.85rem;
                margin-left: 0.5rem;
                background-color: transparent;
                border-color: white;
            }
            .refresh-btn:hover {
                background-color: rgba(255,255,255,0.2);
                border-color: white;
            }
            .btn-primary {
                background-color: #10a37f;
                border-color: #10a37f;
            }
            .btn-primary:hover {
                background-color: #0d8a6f;
                border-color: #0d8a6f;
            }
            .list-group-item {
                border-radius: 6px;
                margin-bottom: 0.5rem;
            }
            pre {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid #10a37f;
            }
            .badge {
                font-weight: 500;
            }
            .bg-primary {
                background-color: #10a37f !important;
            }
            table {
                border-collapse: separate;
                border-spacing: 0;
            }
            table th:first-child {
                border-top-left-radius: 8px;
            }
            table th:last-child {
                border-top-right-radius: 8px;
            }
            .history-row-success {
                background-color: rgba(16, 163, 127, 0.05);
            }
            .history-row-success:hover {
                background-color: rgba(16, 163, 127, 0.1);
            }
            .history-row-error {
                background-color: rgba(220, 53, 69, 0.05);
            }
            .history-row-error:hover {
                background-color: rgba(220, 53, 69, 0.1);
            }
            .model-name {
                font-weight: 500;
                padding: 2px 6px;
                border-radius: 4px;
                display: inline-block;
            }
            .model-claude {
                color: #FF6B00;
            }
            .model-openai {
                color: #000000;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header text-center">
                <h1>OpenAI Code</h1>
                <p class="mb-0">Use OpenAI models with Cursor's Claude Code feature</p>
            </div>

            <!-- Error alert container - will be populated by JavaScript when errors occur -->
            <div id="errorContainer" class="mb-4"></div>

            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header d-flex align-items-center">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-gear-fill config-icon" viewBox="0 0 16 16">
                                <path d="M9.405 1.05c-.413-1.4-2.397-1.4-2.81 0l-.1.34a1.464 1.464 0 0 1-2.105.872l-.31-.17c-1.283-.698-2.686.705-1.987 1.987l.169.311c.446.82.023 1.841-.872 2.105l-.34.1c-1.4.413-1.4 2.397 0 2.81l.34.1a1.464 1.464 0 0 1 .872 2.105l-.17.31c-.698 1.283.705 2.686 1.987 1.987l.311-.169a1.464 1.464 0 0 1 2.105.872l.1.34c.413 1.4 2.397 1.4 2.81 0l.1-.34a1.464 1.464 0 0 1 2.105-.872l.31.17c1.283.698 2.686-.705 1.987-1.987l-.169-.311a1.464 1.464 0 0 1 .872-2.105l.34-.1c1.4-.413 1.4-2.397 0-2.81l-.34-.1a1.464 1.464 0 0 1-.872-2.105l.17-.31c.698-1.283-.705-2.686-1.987-1.987l-.311.169a1.464 1.464 0 0 1-2.105-.872l-.1-.34z"/>
                            </svg>
                            Configuration
                        </div>
                        <div class="card-body">
                            <form id="modelForm" action="/update_models" method="post">
                                <div class="mb-3">
                                    <label for="bigModel" class="form-label">Big Model (for Sonnet)</label>
                                    <select class="form-select" id="bigModel" name="big_model">
                                        {% for model in available_models %}
                                            <option value="{{ model.value }}" {% if model.value == big_model %}selected{% endif %}>{{ model.label }}</option>
                                        {% endfor %}
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label for="smallModel" class="form-label">Small Model (for Haiku)</label>
                                    <select class="form-select" id="smallModel" name="small_model">
                                        {% for model in available_models %}
                                            <option value="{{ model.value }}" {% if model.value == small_model %}selected{% endif %}>{{ model.label }}</option>
                                        {% endfor %}
                                    </select>
                                </div>
                                <div class="reasoning-note mb-3">
                                    <strong>Model Options:</strong>
                                    <ul class="mb-0 mt-2">
                                        <li><strong>claude-3-haiku/sonnet:</strong> Use the original Claude models (requires Anthropic API key)</li>
                                        <li><strong>OpenAI models:</strong> Use gpt-4o, gpt-4o-mini instead of Claude models</li>
                                        <li><strong>Reasoning models:</strong> When using o3-mini or o1, reasoning_effort="medium" is automatically added</li>
                                    </ul>
                                </div>
                                <p><i>Note: The proxy automatically adds reasoning_effort="high" for reasoning models (o3-mini, o1).</i></p>
                                <button type="submit" class="btn btn-primary">Save Configuration</button>
                            </form>
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-header">
                            Connection Info
                        </div>
                        <div class="card-body">
                            <h5>How to connect:</h5>
                            <pre class="bg-light p-3 rounded">ANTHROPIC_BASE_URL=http://localhost:8082 claude</pre>
                            <p>Run this command in your terminal to connect to this proxy and use with Cursor.</p>

                            <h5 class="mt-3">Current Mapping:</h5>
                            <ul class="list-group">
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Claude Sonnet
                                    <span class="badge bg-primary rounded-pill model-badge">{{ big_model }}</span>
                                </li>
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Claude Haiku
                                    <span class="badge bg-primary rounded-pill model-badge">{{ small_model }}</span>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>

                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <div>
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-activity config-icon" viewBox="0 0 16 16">
                                    <path fill-rule="evenodd" d="M6 2a.5.5 0 0 1 .47.33L10 12.036l1.53-4.208A.5.5 0 0 1 12 7.5h3.5a.5.5 0 0 1 0 1h-3.15l-1.88 5.17a.5.5 0 0 1-.94 0L6 3.964 4.47 8.171A.5.5 0 0 1 4 8.5H.5a.5.5 0 0 1 0-1h3.15l1.88-5.17A.5.5 0 0 1 6 2Z"/>
                                </svg>
                                Request History
                            </div>
                            <button id="refreshHistory" class="btn btn-sm btn-outline-light refresh-btn">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-arrow-clockwise" viewBox="0 0 16 16">
                                    <path fill-rule="evenodd" d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2v1z"/>
                                    <path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466z"/>
                                </svg>
                                Refresh
                            </button>
                        </div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-hover" id="historyTable">
                                    <thead>
                                        <tr>
                                            <th>Time</th>
                                            <th>Original</th>
                                            <th>Mapped To</th>
                                            <th>Messages</th>
                                            <th>Status</th>
                                            <th>Error</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {% for req in request_history %}
                                        <tr class="{% if req.status == 'success' %}history-row-success{% else %}history-row-error{% endif %}">
                                            <td>{{ req.timestamp }}</td>
                                            <td><span class="model-name {% if 'claude' in req.original_model.lower() %}model-claude{% else %}model-openai{% endif %}">{{ req.original_model }}</span></td>
                                            <td><span class="model-name {% if 'claude' in req.mapped_model.lower() %}model-claude{% else %}model-openai{% endif %}">{{ req.mapped_model }}</span></td>
                                            <td>{{ req.num_messages }}</td>
                                            <td class="{% if req.status == 'success' %}status-success{% else %}status-error{% endif %}">
                                                {{ req.status }}
                                            </td>
                                            <td>
                                                {% if req.status == 'error' and req.error %}
                                                <span class="badge bg-danger">{{ req.error }}</span>
                                                {% endif %}
                                            </td>
                                        </tr>
                                        {% endfor %}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Form submission via AJAX
            document.getElementById('modelForm').addEventListener('submit', function(e) {
                e.preventDefault();

                const formData = new FormData(this);

                fetch('/update_models', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        // Update UI elements to reflect the change immediately
                        const bigModelBadge = document.querySelector('.list-group-item:nth-child(1) .badge');
                        const smallModelBadge = document.querySelector('.list-group-item:nth-child(2) .badge');

                        if (bigModelBadge) bigModelBadge.textContent = data.big_model;
                        if (smallModelBadge) smallModelBadge.textContent = data.small_model;

                        // Add class for Claude models to style them differently
                        if (bigModelBadge) {
                            if (data.big_model.toLowerCase().includes('claude')) {
                                bigModelBadge.classList.add('model-claude');
                                bigModelBadge.classList.remove('model-openai');
                            } else {
                                bigModelBadge.classList.add('model-openai');
                                bigModelBadge.classList.remove('model-claude');
                            }
                        }

                        if (smallModelBadge) {
                            if (data.small_model.toLowerCase().includes('claude')) {
                                smallModelBadge.classList.add('model-claude');
                                smallModelBadge.classList.remove('model-openai');
                            } else {
                                smallModelBadge.classList.add('model-openai');
                                smallModelBadge.classList.remove('model-claude');
                            }
                        }

                        // Show success message
                        const errorContainer = document.getElementById('errorContainer');
                        const successAlert = document.createElement('div');
                        successAlert.className = 'alert alert-success alert-dismissible fade show';
                        successAlert.innerHTML = `
                            <strong>Success!</strong> Model configuration updated.
                            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                        `;
                        errorContainer.appendChild(successAlert);

                        // Auto-dismiss after 5 seconds
                        setTimeout(() => {
                            successAlert.remove();
                        }, 5000);

                        // Refresh history to see if new requests are using the right model
                        refreshHistoryTable();
                    } else {
                        alert('Error: ' + data.message);
                    }
                })
                .catch(error => {
                    alert('Error: ' + error);
                });
            });

            // Manual refresh history
            document.getElementById('refreshHistory').addEventListener('click', function() {
                refreshHistoryTable();
            });

            // Auto-refresh history table every 10 seconds
            function refreshHistoryTable() {
                fetch('/api/history')
                .then(response => response.json())
                .then(data => {
                    const historyTable = document.getElementById('historyTable').getElementsByTagName('tbody')[0];
                    historyTable.innerHTML = '';

                    data.history.forEach(req => {
                        const row = historyTable.insertRow();
                        row.className = req.status === 'success' ? 'history-row-success' : 'history-row-error';

                        const timeCell = row.insertCell(0);
                        timeCell.textContent = req.timestamp;

                        const originalCell = row.insertCell(1);
                        const originalSpan = document.createElement('span');
                        originalSpan.className = 'model-name ' +
                            (req.original_model.toLowerCase().includes('claude') ? 'model-claude' : 'model-openai');
                        originalSpan.textContent = req.original_model;
                        originalCell.appendChild(originalSpan);

                        const mappedCell = row.insertCell(2);
                        const mappedSpan = document.createElement('span');
                        mappedSpan.className = 'model-name ' +
                            (req.mapped_model.toLowerCase().includes('claude') ? 'model-claude' : 'model-openai');
                        mappedSpan.textContent = req.mapped_model;
                        mappedCell.appendChild(mappedSpan);

                        const messagesCell = row.insertCell(3);
                        messagesCell.textContent = req.num_messages;

                        const statusCell = row.insertCell(4);
                        statusCell.textContent = req.status;
                        statusCell.className = req.status === 'success' ? 'status-success' : 'status-error';

                        // Add the error column
                        const errorCell = row.insertCell(5);
                        if (req.status === 'error' && req.error) {
                            const errorBadge = document.createElement('span');
                            errorBadge.className = 'badge bg-danger';
                            errorBadge.textContent = req.error;
                            errorCell.appendChild(errorBadge);

                            // Make it clickable to show full error
                            errorBadge.style.cursor = 'pointer';
                            errorBadge.addEventListener('click', function() {
                                // Create and show a modal with full error details
                                const errorContainer = document.getElementById('errorContainer');
                                const errorAlert = document.createElement('div');
                                errorAlert.className = 'alert alert-danger alert-dismissible fade show';
                                errorAlert.innerHTML = `
                                    <h5>Error Details:</h5>
                                    <pre>${req.error}</pre>
                                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                                `;
                                errorContainer.appendChild(errorAlert);

                                // Scroll to the top to see the error
                                window.scrollTo(0, 0);
                            });
                        }
                    });
                })
                .catch(error => {
                    console.error('Error refreshing history:', error);
                });
            }

            // Set up auto-refresh
            let autoRefreshInterval = setInterval(refreshHistoryTable, 10000);

            // Initial load of the history table
            refreshHistoryTable();

            // Auto-retry logic for common API errors
            window.addEventListener('error', function(event) {
                // Check if the error is from an API call
                if (event.message && event.message.includes('API')) {
                    // For specific errors like "Anthropic API is overloaded", we'll auto-retry
                    if (event.message.includes('overloaded') ||
                        event.message.includes('rate limit') ||
                        event.message.includes('timeout')) {

                        console.log('API error detected. Will auto-retry in 30 seconds...');
                        // Show a user-friendly message
                        const errorMessage = document.createElement('div');
                        errorMessage.className = 'alert alert-warning alert-dismissible fade show';
                        errorMessage.innerHTML = `
                            <strong>API Error:</strong> ${event.message}
                            <br>Will automatically retry in 30 seconds...
                            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                        `;

                        // Add error to the dedicated error container
                        const errorContainer = document.getElementById('errorContainer');
                        errorContainer.appendChild(errorMessage);

                        // Auto-retry the request after 30 seconds
                        setTimeout(function() {
                            // Remove the error message
                            errorMessage.remove();
                            // Refresh the page to retry
                            window.location.reload();
                        }, 30000);
                    }
                }
            });
        </script>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """

# Make sure the variable is available when importing
__all__ = ['index_html']
