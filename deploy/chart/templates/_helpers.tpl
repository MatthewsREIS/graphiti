{{- define "graphiti-mcp.name" -}}graphiti-mcp{{- end -}}
{{- define "graphiti-mcp.labels" -}}
app.kubernetes.io/name: {{ include "graphiti-mcp.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
