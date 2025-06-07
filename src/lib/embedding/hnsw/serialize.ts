
import { HNSW } from './main';
import { Node } from './node';

const MAGIC = 0x48534e57; // "HNSW" in ASCII

export function encodeGraph(graph: HNSW): Uint8Array {
  // Calculate total size needed
  let totalSize = 14; // header: magic(4) + dim(2) + M(2) + totalNodes(4) + levelMax(2)
  
  for (const [nodeId, node] of graph.nodes) {
    totalSize += 5; // nodeId(4) + levelCount(1)
    for (let level = 0; level <= node.level; level++) {
      totalSize += 2; // neighbourCount(2)
      const neighbors = node.neighbors[level].filter(id => id !== -1);
      totalSize += neighbors.length * 4; // neighbourId(4) each
    }
  }

  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);
  let offset = 0;

  // Write header
  view.setUint32(offset, MAGIC, false); // big-endian for cross-platform consistency
  offset += 4;
  
  view.setUint16(offset, graph.d || 0, false);
  offset += 2;
  
  view.setUint16(offset, graph.M, false);
  offset += 2;
  
  view.setUint32(offset, graph.nodes.size, false);
  offset += 4;
  
  view.setUint16(offset, graph.levelMax, false);
  offset += 2;

  // Write nodes
  for (const [nodeId, node] of graph.nodes) {
    view.setUint32(offset, nodeId, false);
    offset += 4;
    
    view.setUint8(offset, node.level + 1); // levelCount = level + 1
    offset += 1;
    
    for (let level = 0; level <= node.level; level++) {
      const neighbors = node.neighbors[level].filter(id => id !== -1);
      view.setUint16(offset, neighbors.length, false);
      offset += 2;
      
      for (const neighborId of neighbors) {
        view.setUint32(offset, neighborId, false);
        offset += 4;
      }
    }
  }

  return new Uint8Array(buffer);
}

export function decodeGraph(bytes: Uint8Array): HNSW {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 0;

  // Read header
  const magic = view.getUint32(offset, false);
  if (magic !== MAGIC) {
    throw new Error(`Invalid HNSW magic number: 0x${magic.toString(16)}`);
  }
  offset += 4;

  const d = view.getUint16(offset, false);
  offset += 2;
  
  const M = view.getUint16(offset, false);
  offset += 2;
  
  const totalNodes = view.getUint32(offset, false);
  offset += 4;
  
  const levelMax = view.getUint16(offset, false);
  offset += 2;

  // Create HNSW instance
  const graph = new HNSW(M, 200, d || null);
  graph.levelMax = levelMax;

  // Read nodes
  for (let i = 0; i < totalNodes; i++) {
    const nodeId = view.getUint32(offset, false);
    offset += 4;
    
    const levelCount = view.getUint8(offset);
    offset += 1;
    
    const nodeLevel = levelCount - 1;
    
    // Create a placeholder node - we'll need to reconstruct the vector later
    const node = new Node(nodeId, new Float32Array(d || 384), nodeLevel, M);
    
    // Read neighbors for each level
    for (let level = 0; level <= nodeLevel; level++) {
      const neighborCount = view.getUint16(offset, false);
      offset += 2;
      
      const neighbors: number[] = [];
      for (let j = 0; j < neighborCount; j++) {
        const neighborId = view.getUint32(offset, false);
        neighbors.push(neighborId);
        offset += 4;
      }
      
      // Pad with -1 to match expected array size
      while (neighbors.length < M) {
        neighbors.push(-1);
      }
      
      node.neighbors[level] = neighbors;
    }
    
    graph.nodes.set(nodeId, node);
    
    // Set entry point to first node (will be overridden if needed)
    if (graph.entryPointId === -1) {
      graph.entryPointId = nodeId;
    }
  }

  return graph;
}
