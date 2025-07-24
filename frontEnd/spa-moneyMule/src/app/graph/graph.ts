// import { Component } from '@angular/core';
// import { FormsModule } from '@angular/forms';
// import { CommonModule } from '@angular/common';
// import { ActivatedRoute } from '@angular/router';
// import { NgxGraphModule } from '@swimlane/ngx-graph';
// import { Results } from '../results';
// import { NO_ERRORS_SCHEMA } from '@angular/core';

// declare const vis: any;

// @Component({
//   standalone: true,
//   selector: 'app-graph',
//   imports: [FormsModule, CommonModule, NgxGraphModule],
//   schemas: [NO_ERRORS_SCHEMA],
//   templateUrl: './graph.html',
//   styleUrl: './graph.css',
// })
// export class Graph {
//   accountId = '';
//   network: any;
//   visNodes = new vis.DataSet();
//   visEdges = new vis.DataSet();
//   physicsEnabled = true;

//   constructor(private route: ActivatedRoute, private api: Results) {}

//   ngOnInit() {
//     this.route.paramMap.subscribe((params) => {
//       this.accountId = params.get('id') || '';
//       if (this.accountId) {
//         this.fetchGraphData(this.accountId);
//       }
//     });
//   }

//   initializeNetwork() {
//     const container = document.getElementById('mynetwork');
//     if (!container) return;

//     const data = { nodes: this.visNodes, edges: this.visEdges };
//     const options = {
//       nodes: {
//         shape: 'dot',
//         size: 20,
//         font: { size: 16, color: '#333', strokeWidth: 2, strokeColor: '#fff' },
//         borderWidth: 2,
//         shadow: true,
//       },
//       edges: {
//         width: 3,
//         shadow: true,
//         arrows: 'to',
//         color: {
//           color: '#999',
//           highlight: '#1e3a8a',
//           hover: '#1d4ed8',
//           opacity: 0.8,
//         },
//         smooth: { enabled: true, type: 'dynamic', roundness: 0.5 },
//         font: { size: 12, color: '#333', background: '#f8fafc' },
//       },
//       physics: {
//         enabled: this.physicsEnabled,
//         barnesHut: {
//           gravitationalConstant: -10000,
//           centralGravity: 0.05,
//           springLength: 350,
//           springConstant: 0.08,
//           Damping: 0.999,
//           avoidOverlap: 1,
//         },
//         solver: 'barnesHut',
//         stabilization: {
//           enabled: true,
//           iterations: 4000,
//           updateInterval: 25,
//           fit: true,
//         },
//       },
//       interaction: {
//         hover: true,
//         navigationButtons: true,
//         keyboard: true,
//         dragNodes: true,
//         dragView: true,
//         zoomView: true,
//       },
//     };

//     if (this.network) {
//       this.network.setData(data);
//       this.network.setOptions(options);
//     } else {
//       this.network = new vis.Network(container, data, options);

//       this.network.once('stabilizationIterationsDone', () => {
//         this.physicsEnabled = false;
//         this.network.setOptions({ physics: false });
//       });

//       this.network.on('click', (params: { nodes: string | any[]; }) => {
//         if (params.nodes.length) {
//           const node = this.visNodes.get(params.nodes[0]);
//           alert(`Clicked Node: ${node.label} (ID: ${node.id})`);
//         }
//       });
//     }
//   }

//   async fetchGraphData(accountId: string) {
//     try {
//       const data: any = await this.api.fetchGraph(accountId).toPromise();
//       this.visNodes.clear();
//       this.visEdges.clear();
//       this.visNodes.add(data.nodes);
//       this.visEdges.add(data.edges);
//       this.physicsEnabled = true;
//       this.initializeNetwork();
//     } catch (err) {
//       console.error(err);
//       this.visNodes.clear();
//       this.visEdges.clear();
//       this.initializeNetwork();
//     }
//   }

//   togglePhysics() {
//     this.physicsEnabled = !this.physicsEnabled;
//     this.network?.setOptions({ physics: this.physicsEnabled });
//   }
// }

// // export class Graph {
// //   accountId = '';
// //   nodes: any[] = [];
// //   links: any[] = [];

// //   constructor(private route: ActivatedRoute, private api: Results) {}

// //   ngOnInit(): void {
// //     this.accountId = this.route.snapshot.paramMap.get('id')!;
// //     console.log('Fetching graph for:', this.accountId);

// //     this.api.fetchGraph(this.accountId).subscribe({
// //       next: (data: any) => {
// //         console.log('Raw graph data:', data);

// //         // Validate incoming data
// //         if (!data.nodes || !data.links) {
// //           console.error('Graph data missing nodes or links');
// //           return;
// //         }

// //         // Map nodes with default dimensions and colors
// //         this.nodes = data.nodes.map((n: any) => ({
// //           id: n.id,
// //           label: n.label,
// //           data: { color: n.color || '#4e73df' },
// //           dimension: { width: 120, height: 40 },
// //         }));

// //         // Map links, use 'links' array and ensure unique IDs
// //         this.links = data.links.map((e: any, i: number) => ({
// //           id: e.source + '_' + e.target + '_' + i,
// //           source: e.source,
// //           target: e.target,
// //           label: e.type,
// //           color:
// //             e.color || (e.type === 'COLLUSION_PATTERN' ? 'red' : 'steelblue'),
// //         }));

// //         console.log(
// //           `Processed ${this.nodes.length} nodes and ${this.links.length} links`
// //         );
// //       },
// //       error: (err) => console.error('Error fetching graph', err),
// //     });
// //   }
// // }

import { Component, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import axios from 'axios';
import { ActivatedRoute } from '@angular/router';
import { Network } from 'vis-network';
import { DataSet } from 'vis-data';
import type { Options } from 'vis-network';
import { Results } from '../results';

interface GraphResponse {
  nodes: any[];
  links: any[];
  details: any [];
}

@Component({
  standalone: true,
  imports: [CommonModule],
  selector: 'app-graph',
  templateUrl: './graph.html',
  styleUrls: ['./graph.css'],
})
export class Graph {
  @ViewChild('network', { static: false }) networkContainer!: ElementRef;

  graphData: { nodes: any[]; links: any[]; details:any [] } | null = null;
  private network!: Network;
  viewReady = false;
  isLoading = true;
  transactions: any = [];

  constructor(private api: Results, private route: ActivatedRoute) {}

  ngOnInit() {
    this.route.params.subscribe((params) => {
      const id = params['id'];
      if (id) {
        // 1. Get graph structure
        this.api.fetchGraph(id).subscribe({
          next: (data: any) => {
            this.graphData = data;
            console.log(data);
            this.isLoading=false;

            // 2. Get transactions for that account
            this.api.searchAccount(id).subscribe({
              next: () => {
                this.api.fetchResults().subscribe({
                  next: (transactions: any[]) => {
                    // 3. Process and color nodes
                    this.processTransactionsAndColor(transactions);
                    this.tryDrawNetwork();
                  },
                  error: (err: any) =>
                    console.error('Fetch result error:', err),
                });
              },
              error: (err: any) => console.error('Detect error:', err),
            });
          },
          error: (err) => console.error('Graph fetch error:', err),
        });
      }
    });
  }

  ngAfterViewInit() {
    this.viewReady = true;
    this.tryDrawNetwork();
  }

  private tryDrawNetwork() {
    if (this.graphData && this.viewReady && this.networkContainer) {
      this.drawNetwork(this.graphData.nodes, this.graphData.links);
    }
  }

  private drawNetwork(nodes: any[], links: any[]) {
    const data = {
      nodes: new DataSet(
        nodes.map((n) => ({
          id: n.id,
          label: n.label,
          color: n.color, // this allows red/green from processed data
        }))
      ),
      edges: new DataSet(
        links.map((l, i) => ({
          id: `e${i}`,
          from: l.source,
          to: l.target,
          label: l.type,
          arrows: 'to',
        }))
      ),
    };

    const options = { interaction: { hover: true } };

    this.network = new Network(
      this.networkContainer.nativeElement,
      data,
      options
    );
  }

  processTransactionsAndColor(transactions: any[]) {
    const flaggedReceivers = new Map();

    transactions.forEach((tx) => {
      if (tx.Predicted_Fraud_Flag === 1) {
        flaggedReceivers.set(tx.Receiver_Account, 'red');
      } else {
        flaggedReceivers.set(tx.Receiver_Account, 'green');
      }
    });

    // Color nodes based on flagged receivers
    this.graphData!.nodes = this.graphData!.nodes.map((node) => {
      const color = flaggedReceivers.get(node.id);
      return {
        ...node,
        color: color ? { background: color } : undefined,
      };
    });
  }
}
