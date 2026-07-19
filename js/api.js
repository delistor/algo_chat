/**
 * AlgoChat v4 — API Layer
 * Backend communication + SSE streaming + mock support
 */

const API = {
  baseUrl: localStorage.getItem('algochat_api_base') || 'http://localhost:8000',
  mockMode: localStorage.getItem('algochat_mock') === 'true',

  algorithms: [
    { id:'kmeans', name:'KMeans 聚类', description:'K-Means聚类分析', category:'聚类', icon:'🎯',
      input_formats:['.csv','.xlsx'], output_formats:['.csv','.png'],
      params:{ k:{ type:'int', default:3, min:2, max:10, label:'聚类数K' }, max_iter:{ type:'int', default:300, min:50, max:1000, label:'最大迭代' } } },
    { id:'dbscan', name:'DBSCAN 聚类', description:'基于密度的聚类', category:'聚类', icon:'🔵',
      input_formats:['.csv','.xlsx'], output_formats:['.csv','.png'],
      params:{ eps:{ type:'float', default:0.5, min:0.1, max:5.0, step:0.1, label:'邻域半径' }, min_samples:{ type:'int', default:5, min:2, max:20, label:'最小样本数' } } },
    { id:'linear_regression', name:'线性回归', description:'线性回归分析', category:'回归', icon:'📈',
      input_formats:['.csv','.xlsx'], output_formats:['.csv','.png'], params:{} },
    { id:'pca', name:'PCA 降维', description:'主成分分析降维', category:'降维', icon:'🔄',
      input_formats:['.csv','.xlsx'], output_formats:['.csv','.png'],
      params:{ n_components:{ type:'int', default:2, min:1, max:10, label:'目标维度' } } },
    { id:'anomaly_detection', name:'异常检测', description:'基于孤立森林的异常检测', category:'检测', icon:'🔍',
      input_formats:['.csv','.xlsx'], output_formats:['.csv','.png'],
      params:{ contamination:{ type:'float', default:0.1, min:0.01, max:0.5, step:0.01, label:'污染率' } } },
    { id:'data_stats', name:'数据统计', description:'描述性统计分析', category:'统计', icon:'📊',
      input_formats:['.csv','.xlsx','.json'], output_formats:['.csv','.json'], params:{} },
    { id:'image_batch', name:'图片批量处理', description:'批量图片检测与质量分析', category:'检测', icon:'🖼',
      input_formats:['.png','.jpg','.jpeg','.bmp'], output_formats:['.png','.csv','.txt'],
      params:{ threshold:{ type:'float', default:0.85, min:0.5, max:1.0, step:0.01, label:'检测阈值' },
               mode:{ type:'string', default:'标准', label:'处理模式' },
               show_trend:{ type:'string', default:'是', label:'显示趋势图' } } },
  ],

  async getAlgorithms() {
    if (this.mockMode) return this.algorithms;
    try { return await (await fetch(`${this.baseUrl}/api/algorithms`)).json(); }
    catch { return this.algorithms; }
  },

  /**
   * SSE streaming chat with the v4 Agent.
   * Backend sends:  event: <type>\ndata: <json>\n\n
   * Returns an AbortController and an async generator yielding events:
   *   {event: 'content_delta', content: '...'}
   *   {event: 'tool_call', name: '...', args: {...}}
   *   {event: 'tool_result', name: '...', result: ...}
   *   {event: 'done', results: [...]}
   *   {event: 'error', message: '...'}
   */
  streamChat(conversationId, message, files = []) {
    // ── Mock mode: simulate a realistic streaming ReAct conversation ──
    if (this.mockMode) {
      return this._mockStreamChat(conversationId, message, files);
    }

    const controller = new AbortController();

    const generator = (async function* () {
      try {
        const fd = new FormData();
        fd.append('message', message);
        fd.append('conversation_id', conversationId);
        if (files && files.length > 0) {
          files.forEach(f => fd.append('files', f));
        }

        const response = await fetch(`${API.baseUrl}/api/chat/agent`, {
          method: 'POST',
          body: fd,
          signal: controller.signal,
        });

        if (!response.ok) {
          yield { event: 'error', message: `HTTP ${response.status}: ${response.statusText}` };
          return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          // Keep the last incomplete line in buffer
          buffer = lines.pop() || '';

          let currentEvent = 'message';
          let dataLines = [];

          for (const line of lines) {
            // Skip empty lines between events (SSE block separator)
            if (line.trim() === '') {
              // Flush accumulated event
              if (dataLines.length > 0) {
                const parsed = _parseSSEData(currentEvent, dataLines.join('\n'));
                if (parsed) yield parsed;
              }
              currentEvent = 'message';
              dataLines = [];
              continue;
            }

            if (line.startsWith('event: ')) {
              // Save any previously accumulated data before switching event type
              if (dataLines.length > 0) {
                const parsed = _parseSSEData(currentEvent, dataLines.join('\n'));
                if (parsed) yield parsed;
                dataLines = [];
              }
              currentEvent = line.slice(7).trim();
            } else if (line.startsWith('data: ')) {
              dataLines.push(line.slice(6));
            } else if (line.startsWith(':')) {
              // Comment — ignore
            } else {
              // Unprefixed data
              dataLines.push(line.trim());
            }
          }

          // Flush any remaining data from this chunk
          if (dataLines.length > 0) {
            const parsed = _parseSSEData(currentEvent, dataLines.join('\n'));
            if (parsed) yield parsed;
          }
        }
      } catch (e) {
        if (e.name === 'AbortError') return;
        yield { event: 'error', message: e.message };
      }
    })();

    return { controller, generator };
  },

  /**
   * Mock streaming ReAct agent — simulates a full multi-step conversation
   * with tool calls, tool results, and follow-up reasoning.
   * This makes the UI fully interactive even without a backend LLM.
   */
  _mockStreamChat(conversationId, message) {
    const controller = new AbortController();
    const msg = (message || '').toLowerCase();

    const generator = (async function* () {
      await _sleep(400);

      // Step 1: Determine which algorithm to call based on keywords
      let algoId = 'data_stats';
      let algoName = '数据统计';
      let algoIcon = '📊';

      if (msg.includes('聚类') || msg.includes('kmeans')) {
        algoId = 'kmeans'; algoName = 'KMeans 聚类'; algoIcon = '🎯';
      } else if (msg.includes('dbscan')) {
        algoId = 'dbscan'; algoName = 'DBSCAN 聚类'; algoIcon = '🔵';
      } else if (msg.includes('回归') || msg.includes('线性')) {
        algoId = 'linear_regression'; algoName = '线性回归'; algoIcon = '📈';
      } else if (msg.includes('降维') || msg.includes('pca')) {
        algoId = 'pca'; algoName = 'PCA 降维'; algoIcon = '🔄';
      } else if (msg.includes('异常') || msg.includes('检测')) {
        algoId = 'anomaly_detection'; algoName = '异常检测'; algoIcon = '🔍';
      } else if (msg.includes('图片') || msg.includes('批量') || msg.includes('图像')) {
        algoId = 'image_batch'; algoName = '图片批量处理'; algoIcon = '🖼';
      }

      // Intro text
      const intro = `好的！让我使用**${algoName}**来分析你的数据。\n\n`;
      for (const ch of intro) {
        yield { event: 'content_delta', content: ch };
        await _sleep(15);
      }

      // Tool call
      await _sleep(300);
      yield {
        event: 'tool_call',
        name: algoId,
        args: algoId === 'kmeans' ? { k: 3, max_iter: 300 } :
              algoId === 'pca' ? { n_components: 2 } :
              algoId === 'anomaly_detection' ? { contamination: 0.1 } : {},
      };

      // Simulate processing delay
      await _sleep(600);

      // Tool result
      const mockResult = API._mockAlgorithmResult(algoId);
      yield {
        event: 'tool_result',
        name: algoId,
        result: { results: mockResult.results, success: true },
      };

      // Follow-up reasoning
      await _sleep(500);
      const summary = API._getSummaryText(algoId, msg);
      for (const ch of summary) {
        yield { event: 'content_delta', content: ch };
        await _sleep(15);
      }

      yield { event: 'done' };
    })();

    return { controller, generator };
  },

  _mockAlgorithmResult(algoId) {
    const k = 3;
    const names = ['样品_A.png', '样品_B.png', '样品_C.png', '样品_D.png', '样品_E.png'];

    if (algoId === 'image_batch') {
      const results = [];
      results.push({ id:'batch_trend', name:'📈 趋势图', type:'chart', chartType:'line',
        data:{ labels:names.map(n=>n.replace('.png','')),
        datasets:[{label:'质量评分',data:names.map(()=> +(0.7+Math.random()*0.28).toFixed(2))},
                  {label:'阈值',data:names.map(()=>0.85)}] } });
      results.push({ id:'batch_summary', name:'📊 汇总表', type:'table',
        columns:['文件','状态','得分','缺陷数','处理时间'],
        rows:names.map(n=>[n, Math.random()>0.3?'✅ 合格':'⚠️ 异常', (0.7+Math.random()*0.28).toFixed(3), Math.floor(Math.random()*5), (Math.random()*2+0.5).toFixed(2)+'s']) });
      names.forEach(n => {
        results.push({ id:`batch_img_${n}`, name:'处理结果', type:'image', group:n, src:this._mockImageSvg(n,'scatter') });
        results.push({ id:`batch_table_${n}`, name:'检测数值', type:'table', group:n,
          columns:['指标','数值','单位'],
          rows:[['清晰度',(60+Math.random()*40).toFixed(1),'dB'],['噪声',(Math.random()*15).toFixed(2),'%'],['对比度',(0.5+Math.random()*0.5).toFixed(3),''],['亮度',(100+Math.random()*100).toFixed(0),'cd/m²']] });
        results.push({ id:`batch_doc_${n}`, name:'分析摘要', type:'document', group:n,
          content:`## ${n} 分析报告\n\n- **状态**: ${Math.random()>0.3?'合格':'异常'}\n- **质量评分**: ${(0.7+Math.random()*0.28).toFixed(3)}\n- **缺陷区域**: ${Math.floor(Math.random()*3)} 处\n- **建议**: ${Math.random()>0.5?'无需处理':'建议重新采样'}\n\n处理完成于 ${new Date().toLocaleTimeString('zh-CN')}` });
      });
      return { results, success: true };
    }

    return {
      results: [
        { id:`algo_${algoId}_chart`, name:'分析结果图', type:'chart', chartType: algoId==='kmeans'?'scatter':'bar',
          data: algoId==='kmeans'
            ? { datasets: Array.from({length:k},(_,i)=>({ label:`聚类 ${i+1}`, data: Array.from({length:20},()=>({x:(Math.random()-0.5)*4+i*3-3, y:(Math.random()-0.5)*4+i*2-2})) }))}
            : { labels:['Q1','Q2','Q3','Q4'], datasets:[{label:'指标',data:[65,59,80,81]}]} },
        { id:`algo_${algoId}_img`, name:`${algoId}_result.png`, type:'image', src:this._mockImageSvg(`${algoId} 结果`,'scatter') },
        { id:`algo_${algoId}_table`, name:'统计结果表', type:'table',
          columns:['指标','样本数','均值','方差'],
          rows: Array.from({length:k},(_,i)=>[`${i+1}`,`${Math.floor(Math.random()*50+30)}`,`${(Math.random()*6-3).toFixed(2)}`,`${(Math.random()*2+0.5).toFixed(3)}`]) },
        { id:`algo_${algoId}_report`, name:'分析报告', type:'document',
          content:`算法: ${algoId}\n执行时间: 0.82s\n样本总数: ${Math.floor(Math.random()*200+100)}\n\n结论: 数据分析完成，结果如上图表示。` },
      ],
      success: true,
    };
  },

  _getSummaryText(algoId, userMsg) {
    const summaries = {
      kmeans: `\n\n分析完成！数据被成功分为 **3 个聚类**。\n\n从散点图可以看到三个簇之间分离度良好，轮廓系数约 0.72，聚类效果显著。\n\n> 💡 你可以调整聚类数 K 参数来探索不同的分组方式。`,
      dbscan: `\n\nDBSCAN 聚类已完成。\n\n基于密度的聚类发现了若干个自然群组，并将低密度区域的点标记为噪声。这种方式不需要预先指定聚类数量。`,
      linear_regression: `\n\n线性回归分析完成。\n\n模型拟合优度 R² = 0.87，表明自变量对因变量有较强的解释力。可以通过查看回归系数了解各特征的影响方向和程度。`,
      pca: `\n\nPCA 降维完成。\n\n前 2 个主成分累计方差解释率达到 85.3%，说明降维后仍能很好地保留原始数据的信息结构。`,
      anomaly_detection: `\n\n异常检测完成。\n\n基于孤立森林算法，以 ${(Math.random()*10+5).toFixed(1)}% 的污染率识别出了 ${Math.floor(Math.random()*10+3)} 个异常点。建议对这些异常样本进行人工复核。`,
      data_stats: `\n\n数据统计分析完成！\n\n从描述统计表中可以看到各字段的分布概况——均值、标准差和极值范围。如果发现某字段标准差过大，可能存在异常波动，建议进一步排查。`,
      image_batch: `\n\n图片批量处理完成！共处理了 5 张图片。\n\n整体合格率约 ${Math.floor(Math.random()*30+70)}%，建议关注标记为"⚠️ 异常"的样本，可能需要重新采集或检查设备参数。`,
    };
    return summaries[algoId] || summaries.data_stats;
  },

  async sendMessage(conversationId, message, files = []) {
    if (this.mockMode) return this.mockChatResponse(message, files);
    try {
      const fd = new FormData();
      fd.append('message', message);
      fd.append('conversation_id', conversationId);
      files.forEach(f => fd.append('files', f));
      const data = await (await fetch(`${this.baseUrl}/api/chat`, { method:'POST', body:fd })).json();
      this._fixImageUrls(data.results);
      return data;
    } catch (e) { return { type:'error', message:'无法连接后端: ' + e.message }; }
  },

  async runAlgorithm(algoId, files, params = {}) {
    if (this.mockMode) {
      const result = this._mockAlgorithmResult(algoId);
      return { type:'results', message:`算法 **${algoId}** 执行完成`, results: result.results };
    }
    try {
      const fd = new FormData();
      fd.append('algorithm', algoId);
      fd.append('params', JSON.stringify(params));
      files.forEach(f => fd.append('files', f));
      const data = await (await fetch(`${this.baseUrl}/api/algorithm/run`, { method:'POST', body:fd })).json();
      this._fixImageUrls(data.results);
      return data;
    } catch (e) { return { type:'error', message:'算法执行失败: ' + e.message }; }
  },

  async getPreview(fileId) {
    if (this.mockMode) return this.mockPreview(fileId);
    try { return await (await fetch(`${this.baseUrl}/api/files/${fileId}/preview`)).json(); }
    catch { return this.mockPreview(fileId); }
  },

  getDownloadUrl(fileId) { return this.mockMode ? '#' : `${this.baseUrl}/api/files/${fileId}/download`; },

  _fixImageUrls(results) {
    if (!results) return;
    results.forEach(r => {
      if (r.type === 'image' && r.src && !r.src.startsWith('data:')) r.src = this.baseUrl + r.src;
    });
  },

  // ── Mock SVG Generator ──
  _mockImageSvg(title, type) {
    const w=600, h=400, colors=['#7D9E8D','#8E4E26','#992D1E','#A0B9AA','#C49B60'];
    let el='';
    if (type==='scatter') { for(let c=0;c<3;c++) for(let i=0;i<25;i++) el+=`<circle cx="${80+Math.random()*440+c*30}" cy="${60+Math.random()*260}" r="5" fill="${colors[c]}" opacity="0.7" stroke="white" stroke-width="0.5"/>`; }
    else if (type==='line') { for(let d=0;d<2;d++){let p=[];for(let i=0;i<12;i++)p.push(`${80+i*38},${300-(40+Math.random()*180)}`);el+=`<polyline points="${p.join(' ')}" fill="none" stroke="${colors[d]}" stroke-width="2.5"/>`;} }
    else { for(let i=0;i<5;i++){const bh=60+Math.random()*180;el+=`<rect x="${80+i*90}" y="${300-bh}" width="50" height="${bh}" fill="${colors[i]}" opacity="0.8" rx="3"/>`;} }
    return 'data:image/svg+xml;base64,'+btoa(unescape(encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}"><rect width="${w}" height="${h}" fill="#FAFCF5"/><text x="${w/2}" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#3C2819">${title}</text><line x1="70" y1="300" x2="560" y2="300" stroke="#D4C5A0" stroke-width="1"/><line x1="70" y1="50" x2="70" y2="300" stroke="#D4C5A0" stroke-width="1"/>${el}</svg>`)));
  },

  // ── Mock Responses (non-streaming fallback) ──
  mockChatResponse(msg) {
    const m = msg.toLowerCase();
    if (m.includes('图表')||m.includes('可视化')||m.includes('chart')) return this.mockChartResult();
    if (m.includes('表格')||m.includes('数据')||m.includes('table')) return this.mockTableResult();
    if (m.includes('聚类')||m.includes('kmeans')) return this.mockAlgorithmResponse('kmeans');
    if (m.includes('批量')||m.includes('图片处理')||m.includes('image_batch')) return this.mockImageBatchResult();
    if (m.includes('分析')||m.includes('统计')) return this.mockAnalysisResult();
    return { type:'chat', message:'你好！我是 AlgoChat 助手。\n\n你可以：\n- 直接与我对话\n- 在左侧选择算法\n- 上传文件并执行算法\n\n试试输入「图表」「表格」「分析」，或在左侧点击算法开始。' };
  },

  mockChartResult() {
    return { type:'results', message:'以下是数据可视化结果：', results:[
      { id:'chart_1', name:'趋势分析图', type:'chart', chartType:'line',
        data:{ labels:['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],
          datasets:[{label:'销售额',data:[65,59,80,81,56,55,72,85,90,78,88,95]},{label:'成本',data:[28,32,40,45,38,35,42,50,55,48,52,58]}]} },
      { id:'chart_1_img', name:'趋势分析图.png', type:'image', src:this._mockImageSvg('趋势分析图','line') },
      { id:'chart_2', name:'分类分布饼图', type:'chart', chartType:'pie',
        data:{ labels:['A类','B类','C类','D类'], datasets:[{data:[35,25,22,18]}]} },
    ]};
  },

  mockTableResult() {
    return { type:'results', message:'以下是数据表格结果：', results:[
      { id:'table_1', name:'统计数据.csv', type:'table',
        columns:['指标','Q1','Q2','Q3','Q4'],
        rows:[['营收','1200','1350','1580','1720'],['利润','340','380','450','520'],['增长率','8.2%','12.5%','17.0%','8.9%'],['客户数','1520','1780','2100','2450']] },
    ]};
  },

  mockAlgorithmResponse(algoId, params={}) {
    if (algoId === 'image_batch') return this.mockImageBatchResult(params);
    const result = this._mockAlgorithmResult(algoId);
    const k = params.k || 3;
    return { type:'results', message:`算法 **${algoId}** 执行完成，共发现 ${k} 个聚类。`, results: result.results };
  },

  mockAnalysisResult() {
    return { type:'results', message:'统计分析完成，结果如下：', results:[
      { id:'stats_chart', name:'月度趋势', type:'chart', chartType:'bar',
        data:{ labels:['1月','2月','3月','4月','5月','6月'], datasets:[{label:'均值',data:[42,38,55,47,60,53]},{label:'中位数',data:[40,36,52,45,58,50]}]} },
      { id:'stats_img', name:'统计分析图.png', type:'image', src:this._mockImageSvg('统计分析','bar') },
      { id:'stats_table', name:'描述统计.csv', type:'table',
        columns:['字段','均值','标准差','最小值','最大值','中位数'],
        rows:[['温度','23.5','4.2','12.1','35.8','23.0'],['湿度','65.2','12.8','30.0','98.0','64.0'],['气压','1013.2','8.5','990.1','1035.7','1013.0'],['风速','3.8','2.1','0.2','15.6','3.2']] },
    ]};
  },

  mockImageBatchResult(params={}) {
    const names = ['样品_A.png','样品_B.png','样品_C.png','样品_D.png','样品_E.png'];
    const results = [];
    results.push({ id:'batch_trend', name:'📈 趋势图', type:'chart', chartType:'line',
      data:{ labels:names.map(n=>n.replace('.png','')),
        datasets:[{label:'质量评分',data:names.map(()=> +(0.7+Math.random()*0.28).toFixed(2))},
                  {label:'阈值',data:names.map(()=>0.85)}] } });
    results.push({ id:'batch_summary', name:'📊 汇总表', type:'table',
      columns:['文件','状态','得分','缺陷数','处理时间'],
      rows:names.map(n=>[n, Math.random()>0.3?'✅ 合格':'⚠️ 异常', (0.7+Math.random()*0.28).toFixed(3), Math.floor(Math.random()*5), (Math.random()*2+0.5).toFixed(2)+'s']) });
    names.forEach(n => {
      results.push({ id:`batch_img_${n}`, name:'处理结果', type:'image', group:n, src:this._mockImageSvg(n,'scatter') });
      results.push({ id:`batch_table_${n}`, name:'检测数值', type:'table', group:n,
        columns:['指标','数值','单位'],
        rows:[['清晰度',(60+Math.random()*40).toFixed(1),'dB'],['噪声',(Math.random()*15).toFixed(2),'%'],['对比度',(0.5+Math.random()*0.5).toFixed(3),''],['亮度',(100+Math.random()*100).toFixed(0),'cd/m²']] });
      results.push({ id:`batch_doc_${n}`, name:'分析摘要', type:'document', group:n,
        content:`## ${n} 分析报告\n\n- **状态**: ${Math.random()>0.3?'合格':'异常'}\n- **质量评分**: ${(0.7+Math.random()*0.28).toFixed(3)}\n- **缺陷区域**: ${Math.floor(Math.random()*3)} 处\n- **建议**: ${Math.random()>0.5?'无需处理':'建议重新采样'}\n\n处理完成于 ${new Date().toLocaleTimeString('zh-CN')}` });
    });
    return { type:'results', message:`图片批量处理完成，共处理 ${names.length} 张图片。`, results };
  },

  mockPreview(fileId) {
    if (fileId.includes('table')) return { type:'table', columns:['ID','名称','数值','状态'], rows:[['1','样本A','23.5','正常'],['2','样本B','45.2','异常'],['3','样本C','12.8','正常']] };
    if (fileId.includes('chart')) return this.mockChartResult().results.find(r=>r.id===fileId)||{type:'chart',chartType:'line',data:{labels:['A','B','C','D'],datasets:[{label:'值',data:[10,20,15,25]}]}};
    if (fileId.includes('img')) return {type:'image',name:'result.png',src:this._mockImageSvg('预览图','scatter')};
    return {type:'document',content:'这是文档预览内容。\n\n算法分析报告\n━━━━━━━━━━\n处理状态: 完成\n样本数: 150\n特征数: 4'};
  },
};

// ── Private helpers ──

/**
 * Parse SSE data payload into a typed event object.
 * Maps the raw backend JSON to the frontend event schema.
 */
function _parseSSEData(eventName, dataStr) {
  if (!dataStr) return null;

  let parsed;
  try {
    parsed = JSON.parse(dataStr);
  } catch {
    // Non-JSON data — treat as raw text content
    if (eventName === 'content_delta') {
      return { event: 'content_delta', content: dataStr };
    }
    return null;
  }

  // Backend sends: {"event": "content_delta", "content": "..."} etc.
  // Map to frontend event names
  const evt = parsed.event || eventName;

  switch (evt) {
    case 'content_delta':
      return { event: 'content_delta', content: parsed.content || parsed.delta || '' };
    case 'tool_call':
      return { event: 'tool_call', name: parsed.name || '', args: parsed.args || parsed.arguments || {} };
    case 'tool_result':
      return {
        event: 'tool_result',
        name: parsed.name || '',
        results: parsed.results || [],
        success: parsed.success || false,
        error: parsed.error || null,
        result: parsed.result || parsed.output || { results: parsed.results || [], success: parsed.success || false },
      };
    case 'done':
      return { event: 'done', message: parsed.message || '', results: parsed.results || [] };
    case 'conversation_id':
      return { event: 'conversation_id', conversation_id: parsed.conversation_id || '' };
    case 'error':
      return { event: 'error', message: parsed.message || dataStr };
    default:
      // Try to infer from payload shape
      if (parsed.content || parsed.delta) {
        return { event: 'content_delta', content: parsed.content || parsed.delta };
      }
      if (parsed.name && parsed.args) {
        return { event: 'tool_call', name: parsed.name, args: parsed.args };
      }
      if (parsed.name && parsed.result) {
        return { event: 'tool_result', name: parsed.name, result: parsed.result };
      }
      if (parsed.message && parsed.type === 'error') {
        return { event: 'error', message: parsed.message };
      }
      return null;
  }
}

function _sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}